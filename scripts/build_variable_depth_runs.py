#!/usr/bin/env python3
"""Build variable-depth retrieval runs per the organisers' 2026 clarification.

THE RULE (organisers, 2026-07-27)
---------------------------------
"For each narrative, you must choose how many documents to return, under the condition that the
document is relevant to the narrative and is useful evidence for producing a high-quality answer
to the narrative. For a given narrative, return all relevant documents and nothing more!"

Two consequences our shipped runs violate:
  * Depth must be chosen PER NARRATIVE. Our runs return a flat 1000; a flat 20 or 30 would be just
    as non-compliant, since neither is variable.
  * Padding is penalised ("nothing more").

WHY THE SHIPPED RUNS CANNOT SIMPLY BE TRUNCATED
-----------------------------------------------
The shipped TSV score column is SYNTHETIC and positional (1000, 999, ... 801 at rank 200, 800 at
rank 201, ... 701 at rank 300, then real BM25 scores in the tail), i.e. the file is a concatenation
of segments. Measured against it, our own cited evidence sits at MEDIAN RANK 218-247 and only
8-18% of it falls inside the top 20. So truncating the shipped order at 20/30 would keep documents
our own generator declined to cite while discarding 82-92% of the evidence that actually produced
our answers, and would desynchronise the R run from the RAG run.

THE POLICIES BUILT HERE
-----------------------
Real relevance scores DO exist in cache: facet_rankings.json maps each facet (and "_anchor") to
[docid, cross_encoder_score]. All policies below use those, so the emitted score column is a true
monotone relevance score — which also matters because trec_eval sorts by SCORE, not by rank, so a
hand-forced order with a non-monotone score column would simply be ignored.

  cited     "A": only the documents our answer actually cites. The most literal reading of
            "useful evidence for producing a high-quality answer". ~10 docs/topic.
  ce_floor  "V2" (recommended): every document scoring at least as high as the WEAKEST document
            our own answer relied on -- floor = min(CE) over the cited set. Self-calibrating per
            narrative, needs no external tuning, and every cited document is a member BY
            DEFINITION (its score >= the min of the cited scores), so ordering purely by CE
            descending keeps the score monotone AND retains 100% of our evidence.
            Measured depth: median 100, range 18-456.
  ce_floor_cap100  V2 truncated to at most 100/topic, as a hedge in case very deep topics are
            penalised. 100 is also the depth the organisers' own retrieval baseline ships.

Writes TREC 6-column runs (topic Q0 docid rank score run_id). Touches no submission.

Usage:
    python scripts/build_variable_depth_runs.py --dry-run
    python scripts/build_variable_depth_runs.py
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "rag" / "cache" / "agentic_v2"
OUT = ROOT / "runs" / "rag" / "variable_depth_candidates"

# submission dir -> (run_id, default cache pool, optional per-topic pool map source)
SYSTEMS = {
    "bp_agentic_v2":     ("bp_agentic_v2",     "official119",       None),
    "bp_probe_guided":   ("bp_probe_guided",   "official119_probe", None),
    "bp_support_argmax": ("bp_support_argmax", "official119",
                          ROOT / "runs/rag/best_of_support_aware_official119.jsonl"),
}
POOL_FOR_SOURCE = {"v2_122b": "official119", "probe": "official119_probe",
                   "v3_loop": "official119"}


def best_ce(topic_dir: Path) -> dict:
    """docid -> best cross-encoder score across all facets and the anchor list."""
    p = topic_dir / "facet_rankings.json"
    if not p.exists():
        return {}
    best = {}
    for _facet, lst in json.load(open(p)).items():
        for docid, sc in lst:
            if docid not in best or sc > best[docid]:
                best[docid] = sc
    return best


def policies(ce: dict, refs: list) -> dict:
    """Return {policy_name: [(docid, score), ...] sorted by score desc}."""
    cited_scores = [ce[d] for d in refs if d in ce]
    out = {}

    # A — cited only
    out["cited"] = sorted(((d, ce.get(d, 0.0)) for d in refs),
                          key=lambda x: -x[1])

    def with_floor(floor):
        keep = {d: s for d, s in ce.items() if s >= floor}
        for d in refs:            # union the cited set back in -> 100% evidence retention always
            keep.setdefault(d, ce.get(d, floor))
        return sorted(keep.items(), key=lambda x: -x[1])

    # V2 — self-calibrated floor at the weakest cited document.
    out["ce_floor"] = with_floor(min(cited_scores)) if cited_scores else out["cited"]

    # V2r — ROBUST floor: trim the single weakest citation before setting the floor. Measured
    # sensitivity: the weakest citation alone DOUBLES the depth (median 100 -> 51), so trimming it
    # halves the padding while the union above still guarantees 100% evidence retention. Better
    # aligned with "and nothing more".
    if len(cited_scores) > 1:
        out["ce_floor_robust"] = with_floor(sorted(cited_scores)[1])
    else:
        out["ce_floor_robust"] = out["ce_floor"]

    out["ce_floor_cap100"] = out["ce_floor"][:100]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    topics = [l.split("\t")[0].strip()
              for l in open(ROOT / "data/rag/topics/trec_rag_2026_queries.tsv") if l.strip()]
    summary = []

    for sub, (run_id, default_pool, map_src) in SYSTEMS.items():
        rag = ROOT / "submissions" / sub / "rag_output_trec_rag_2026.jsonl"
        answers = {str(json.loads(l)["metadata"]["narrative_id"]): json.loads(l)
                   for l in open(rag)}

        # per-topic pool for the mosaic run
        pool_of = {}
        if map_src and map_src.exists():
            for r in map(json.loads, open(map_src)):
                m = r["metadata"]
                pool_of[str(m["topic_id"])] = POOL_FOR_SOURCE.get(
                    m.get("selected_from"), default_pool)

        acc = {p: [] for p in ("cited", "ce_floor", "ce_floor_robust", "ce_floor_cap100")}
        depths = {p: [] for p in acc}
        retained = {p: [0, 0] for p in acc}   # [cited kept, cited total]

        for tid in topics:
            rec = answers.get(tid)
            if rec is None:
                sys.exit(f"FATAL: {sub} missing topic {tid}")
            pool = pool_of.get(tid, default_pool)
            ce = best_ce(CACHE / pool / tid)
            refs = rec.get("references", [])
            if not ce:
                sys.exit(f"FATAL: no facet_rankings for {pool}/{tid} — cannot score.")
            pol = policies(ce, refs)
            for name, ranked in pol.items():
                depths[name].append(len(ranked))
                got = {d for d, _ in ranked}
                retained[name][0] += sum(1 for d in refs if d in got)
                retained[name][1] += len(refs)
                for i, (docid, sc) in enumerate(ranked, 1):
                    acc[name].append(f"{tid} Q0 {docid} {i} {sc:.6f} {run_id}")

        for name in acc:
            d = depths[name]
            r_kept, r_tot = retained[name]
            summary.append({
                "system": sub, "policy": name,
                "median": st.median(d), "mean": round(st.mean(d), 1),
                "min": min(d), "max": max(d), "total_rows": len(acc[name]),
                "cited_retained_pct": round(100 * r_kept / r_tot, 1),
            })
            if not a.dry_run:
                dst = OUT / f"{sub}__{name}"
                dst.mkdir(parents=True, exist_ok=True)
                (dst / "r_output_trec_rag_2026.tsv").write_text("\n".join(acc[name]) + "\n")

    hdr = f"{'system':<20}{'policy':<18}{'median':>7}{'mean':>8}{'min':>6}{'max':>6}{'rows':>9}{'cited kept':>12}"
    print(hdr); print("-" * len(hdr))
    for s in summary:
        print(f"{s['system']:<20}{s['policy']:<18}{s['median']:>7.0f}{s['mean']:>8.1f}"
              f"{s['min']:>6}{s['max']:>6}{s['total_rows']:>9}{s['cited_retained_pct']:>11.1f}%")
    print(f"\nshipped baseline for reference: 1000 docs/topic, 119000 rows, cited kept 100%"
          f" but median cited rank 218-247")
    if a.dry_run:
        print("\nDRY RUN — nothing written.")
    else:
        print(f"\nwrote runs under {OUT}")


if __name__ == "__main__":
    main()
