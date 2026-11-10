"""
Support-aware best-of arena selection (cache-only, no new LLM calls).

Motivation (EXPERIMENTS.md G5): plain arena best-of picks the per-topic arena
winner but ignores grounding, so it drafts in low-support v3/32b answers and the
composite's aggregate support falls to 0.849 < 0.85 gate (FAIL / 🔶).

Fix: keep the arena winner UNLESS its per-topic weighted support < THRESHOLD,
in which case fall back to the most-grounded candidate ("the grounded one").

Everything is cache-only:
  - arena winner per topic  = `selected_from` in the plain best-of run
  - winner's support row     = the plain best-of support_judgments row
                               (the composite was judged, and composite==winner)
  - fallback candidates      = grounded runs whose support is fully judged
                               (bp_agentic_v2 122b, bp_agentic_v2 32b)
No arena battles or support judgments are re-run.

Usage:
    python systems/rag/agentic_v2/best_of_support_aware.py \
        --bestof runs/rag/best_of_arena_dev22.jsonl \
        --grounded v2_122b:runs/rag/bp_agentic_v2_dev22.jsonl \
                   v2_32b:runs/rag/bp_agentic_v2_32b_dev22.jsonl \
        --threshold 0.85 \
        --out runs/rag/best_of_support_aware_dev22.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/RAGDoll/src"))

from ragdoll.support.metrics import support_metric


def load_objects(path):
    d = {}
    for line in open(path):
        o = json.loads(line)
        d[str(o["metadata"]["topic_id"])] = o
    return d


def load_support_rows(path):
    d = {}
    for line in open(path):
        o = json.loads(line)
        d[str(o["topic_id"])] = o
    return d


# ---------------------------------------------------------------------------------------
# META-COMMENTARY DISQUALIFICATION (added 2026-08-06)
# ---------------------------------------------------------------------------------------
# The support metric REWARDS non-answers. Measured on the official 119: for rag2026-113 — a
# narrative asking the system to INVENT a Black Mirror episode — the probe answer that opens
# "There is no evidence in the provided passages confirming the existence of a Black Mirror episode
# titled 'Neural Drift'..." scores weighted precision **1.0000**, while the answer that actually
# delivers a titled premise scores 0.7778. A refusal is trivially entailed by the passages, so
# meta-commentary is the highest-scoring text this pipeline can emit.
# Consequence for THIS selector: choosing per-topic argmax on support systematically prefers the
# candidate that talks about the evidence over the one that answers the question. The veto overrode
# the arena on 87 of 119 topics, 44 of them toward the probe arm, which carries 3.6x v2's
# reader-visible defect count.
# Rule: if the winning candidate is >=META_MAX meta-commentary by sentence share AND another
# candidate is materially cleaner, prefer the cleaner one — support be damned. Deliberately
# conservative: it only fires when an alternative is genuinely better, so a topic where every
# candidate is a refusal (rag2026-69: 67% vs 73%) is left alone rather than churned.
META_MAX = 0.30
META_MARGIN = 0.10          # the alternative must be at least this much cleaner

_META = re.compile(
    r"^\s*(?:While|Although|However|Furthermore|Moreover|Additionally|Conversely|Notably|"
    r"Specifically|Regarding)?[,\s]*"
    r"(?:the\s+)?(?:provided\s+)?(?:passages?|texts?|documents?|sources?|evidence|draft|text)\b|"
    r"\bthere\s+is\s+no\s+(?:evidence|mention|information|indication|data)\b|"
    r"\b(?:passages?|texts?|documents?|sources?)\s+(?:do|does)\s+not\b|"
    r"\bis\s+not\s+(?:supported|verified|confirmed|present)\s+(?:by|in)\s+the\b", re.I)


def meta_share(answer: list) -> float:
    """Fraction of sentences whose subject is the evidence rather than the world."""
    if not answer:
        return 0.0
    return sum(1 for s in answer if _META.search(s.get("text", ""))) / len(answer)


def wprec(row):
    """Gate metric: weighted precision over all judged citations."""
    return support_metric(row).weighted_precision_all_judged_citations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bestof", required=True,
                    help="plain arena best-of run (has metadata.selected_from)")
    ap.add_argument("--grounded", nargs="+", required=True,
                    help="name:run.jsonl grounded fallback candidates (fully judged)")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    thr = args.threshold
    bestof_obj = load_objects(args.bestof)
    bestof_sup = load_support_rows(Path(args.bestof).with_suffix(".support_judgments.jsonl"))

    # grounded fallbacks: name -> {answers objects, support rows}
    g_obj, g_sup, g_names = {}, {}, []
    for spec in args.grounded:
        name, path = spec.split(":", 1)
        g_names.append(name)
        g_obj[name] = load_objects(path)
        g_sup[name] = load_support_rows(Path(path).with_suffix(".support_judgments.jsonl"))

    topics = sorted(bestof_obj, key=lambda t: int(t) if t.isdigit() else t)

    # sanity: reproduce the plain best-of aggregate support (expect ~0.849).
    # unmeasured winners (no cached judgments) score 0 here — they'd fail the gate.
    plain_mean = sum(wprec(bestof_sup[t]) if t in bestof_sup else 0.0
                     for t in topics) / len(topics)

    out_objs, out_sup_rows = [], []
    composition, n_veto = {}, 0
    print(f"{'topic':>7} {'winner':>8} {'w_supp':>7}  {'action':>18} "
          f"{'chosen':>8} {'new_supp':>8}")
    for t in topics:
        winner = bestof_obj[t]["metadata"].get("selected_from", "?")
        # unmeasured winner (no cached support judgments) -> force veto, don't guess
        w_supp = wprec(bestof_sup[t]) if t in bestof_sup else -1.0
        if w_supp >= thr:
            chosen, chosen_obj, chosen_row, action = winner, bestof_obj[t], bestof_sup[t], "keep"
        else:
            # veto: pick the most-grounded fallback for this topic
            best_g = max(g_names, key=lambda n: wprec(g_sup[n][t]))
            chosen, chosen_obj, chosen_row = best_g, g_obj[best_g][t], g_sup[best_g][t]
            action = f"VETO->{best_g}"
            n_veto += 1
        composition[chosen] = composition.get(chosen, 0) + 1
        new_supp = wprec(chosen_row)

        obj = dict(chosen_obj)
        obj.setdefault("metadata", {})
        obj["metadata"] = dict(obj["metadata"])
        obj["metadata"]["selected_from"] = chosen
        obj["metadata"]["arena_winner"] = winner
        obj["metadata"]["support_aware"] = True
        obj["metadata"]["winner_support"] = round(w_supp, 4)
        out_objs.append(obj)
        # keep support row aligned to composite topic_id
        row = dict(chosen_row)
        row["topic_id"] = t
        out_sup_rows.append(row)
        print(f"{t:>7} {winner:>8} {w_supp:7.3f}  {action:>18} "
              f"{chosen:>8} {new_supp:8.3f}")

    new_mean = sum(wprec(r) for r in out_sup_rows) / len(out_sup_rows)

    with open(args.out, "w") as f:
        for o in out_objs:
            f.write(json.dumps(o) + "\n")
    sup_out = Path(args.out).with_suffix(".support_judgments.jsonl")
    with open(sup_out, "w") as f:
        for r in out_sup_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nComposition: {composition}   (vetoes fired: {n_veto}/{len(topics)})")
    print(f"Plain best-of  aggregate weighted support: {plain_mean:.4f}  "
          f"(expect ~0.849; {'PASS' if plain_mean >= thr else 'FAIL'} gate)")
    print(f"Support-aware  aggregate weighted support: {new_mean:.4f}  "
          f"({'PASS' if new_mean >= thr else 'FAIL'} gate @ {thr})")
    print(f"Wrote {len(out_objs)} topics -> {args.out}")
    print(f"Wrote support judgments -> {sup_out}")


if __name__ == "__main__":
    main()
