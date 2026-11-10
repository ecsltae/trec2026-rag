"""
Build a spec-compliant TREC RAG 2026 submission from a best-of COMPOSITE jsonl.

build_submission.py reads per-topic cache dirs (final_ranking.json + answer.json),
so it cannot consume a best-of composite (output of best_of_support_aware.py), which
is a single jsonl of already-selected per-topic answers. This tool closes that gap.

Composite record shape (per line):
  metadata: {topic_id, run_id, type, selected_from, ...}
  references: [docid, ...]                       # docid strings
  answer:     [{text, citations:[int, ...]}]     # citations = 0-based idx into references

Output (matches build_submission.py byte-for-byte in format):
  rag_output_trec_rag_2026.jsonl  — metadata{team_id,run_id,type,narrative_id,title,narrative}
                                    + references(deduped cited docids, first-appearance order)
                                    + answer(citations re-indexed into references)
  r_output_trec_rag_2026.tsv      — 6-col; per topic the ranking comes from the SELECTED
                                    candidate's cache dir (--ranking-map name:topic_set,
                                    else --default-topic-set). Skipped with --skip-tsv.

Usage:
  python systems/rag/agentic_v2/build_submission_from_composite.py \
    --composite runs/rag/best_of_support_aware_official119.jsonl \
    --team-id BiTeM --run-id bp_bestof_support_aware \
    --topics data/rag/topics/trec_rag_2026_queries.tsv \
    --ranking-map probe:official119_probe --default-topic-set official119 \
    --out-dir submissions/bp_bestof_support_aware
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from trec_shared.trec_utils import load_topics, validate_trec_run

CACHE_ROOT = ROOT / "data/rag/cache/agentic_v2"


def official_meta(narrative: str) -> tuple:
    """(title, narrative) with build_submission's title convention: first ~12 words."""
    return " ".join(narrative.split()[:12]), narrative


def build_jsonl(composite: Path, team_id: str, run_id: str, run_type: str,
                topics: dict, out_path: Path) -> tuple:
    n, empty, orphan, bad_idx = 0, 0, 0, 0
    with open(out_path, "w") as f:
        for line in open(composite):
            o = json.loads(line)
            tid = str(o["metadata"]["topic_id"])
            src_refs = o["references"]

            # map int citations back to docids, then re-derive refs exactly like
            # build_submission (deduped cited docids in first-appearance order).
            refs, seen, sentences = [], set(), []
            for s in o["answer"]:
                docids = []
                for c in s["citations"]:
                    if not isinstance(c, int) or c < 0 or c >= len(src_refs):
                        bad_idx += 1
                        continue
                    docids.append(src_refs[c])
                for d in docids:
                    if d not in seen:
                        seen.add(d)
                        refs.append(d)
                sentences.append({"text": s["text"],
                                  "citations": [refs.index(d) for d in docids]})

            cited = {c for s in sentences for c in s["citations"]}
            if len(cited) != len(refs):
                orphan += 1
            if not sentences:
                empty += 1

            t = topics.get(tid)
            narrative = t["narrative"] if t else ""
            title, narrative = official_meta(narrative)
            obj = {
                "metadata": {
                    "team_id": team_id, "run_id": run_id, "type": run_type,
                    "narrative_id": tid, "title": title, "narrative": narrative,
                },
                "references": refs,
                "answer": sentences,
            }
            f.write(json.dumps(obj) + "\n")
            n += 1
    return n, empty, orphan, bad_idx


def build_tsv(composite: Path, run_id: str, ranking_map: dict,
              default_ts: str, out_path: Path) -> tuple:
    """Per topic, pull final_ranking.json from the selected candidate's cache dir."""
    rows, missing = [], []
    for line in open(composite):
        o = json.loads(line)
        tid = str(o["metadata"]["topic_id"])
        sel = o["metadata"].get("selected_from", "")
        ts = ranking_map.get(sel, default_ts)
        fr = CACHE_ROOT / ts / tid / "final_ranking.json"
        if not fr.exists():
            missing.append((tid, sel, ts))
            continue
        for rank, (docid, score) in enumerate(json.loads(fr.read_text()), 1):
            rows.append(f"{tid} Q0 {docid} {rank} {score:.6f} {run_id}")
    out_path.write_text("\n".join(rows) + "\n")
    return len(rows), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composite", required=True)
    ap.add_argument("--team-id", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--type", default="automatic", choices=["automatic", "manual"])
    ap.add_argument("--topics", required=True)
    ap.add_argument("--ranking-map", nargs="*", default=[],
                    help="selected_from:cache_topic_set pairs, e.g. probe:official119_probe")
    ap.add_argument("--default-topic-set", default="official119")
    ap.add_argument("--skip-tsv", action="store_true",
                    help="build only the RAG jsonl (for testing without cache rankings)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    composite = ROOT / args.composite if not Path(args.composite).is_absolute() else Path(args.composite)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = load_topics(args.topics)
    ranking_map = dict(p.split(":", 1) for p in args.ranking_map)

    jsonl_path = out_dir / "rag_output_trec_rag_2026.jsonl"
    n, empty, orphan, bad_idx = build_jsonl(composite, args.team_id, args.run_id,
                                            args.type, topics, jsonl_path)
    print(f"RAG-> {jsonl_path.name}: {n} topics, empty={empty}, "
          f"orphan_refs={orphan}, bad_citation_idx={bad_idx}")

    if not args.skip_tsv:
        tsv_path = out_dir / "r_output_trec_rag_2026.tsv"
        n_rows, missing = build_tsv(composite, args.run_id, ranking_map,
                                    args.default_topic_set, tsv_path)
        errors = validate_trec_run(str(tsv_path), max_per_topic=100000)
        print(f"R  -> {tsv_path.name}: {n_rows} rows, "
              f"{'OK' if not errors else errors[:3]}")
        if missing:
            print(f"  WARNING: {len(missing)} topics missing final_ranking.json "
                  f"(first 3: {missing[:3]})")

    ok = (empty == 0 and orphan == 0 and bad_idx == 0)
    print(f"  {'✓ RAG jsonl spec-compliant' if ok else '✗ PROBLEMS — see counts above'}")


if __name__ == "__main__":
    main()
