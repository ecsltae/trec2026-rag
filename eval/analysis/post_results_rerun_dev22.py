"""
Paired comparison of the writer arms on dev22, as pre-registered in PREREG_post_results_rerun.md.

Reads the summaries written by eval/nugget_local_judge.py --out and prints, for every run against
the shipped answers: mean of strict_all / strict_vital, paired Δ, SE of the paired differences, Δ/SE,
and median words. The primary cell is evidence vs shipped on strict_all; rerun vs shipped is the
null control.

Usage:
    python eval/analysis/post_results_rerun_dev22.py \
        eval/analysis/nugget_dev22_existing.json eval/analysis/nugget_dev22_arms.json
"""
import json
import statistics
import sys

BASE = "bp_agentic_v2_dev22"
METRICS = ("strict_all_score", "strict_vital_score")


def main():
    runs = {}
    for path in sys.argv[1:]:
        runs.update(json.load(open(path))["runs"])
    base = runs[BASE]["per_topic"]
    print(f"{'run':34s} {'n':>3s}  " + "  ".join(f"{m[:-6]:>13s}  {'Δ':>7s} {'SE':>6s} {'Δ/SE':>5s}" for m in METRICS)
          + "  words")
    for name, r in runs.items():
        common = sorted(set(r["per_topic"]) & set(base))
        cells = []
        for m in METRICS:
            mean = statistics.mean(r["per_topic"][t][m] for t in common)
            diffs = [r["per_topic"][t][m] - base[t][m] for t in common]
            delta = statistics.mean(diffs)
            se = statistics.stdev(diffs) / len(diffs) ** 0.5 if name != BASE else 0.0
            flag = "*" if se and abs(delta) > 2 * se else " "
            cells.append(f"{mean:13.4f}  {delta:+7.4f} {se:6.4f} {delta / se if se else 0:5.1f}{flag}")
        print(f"{name:34s} {len(common):3d}  " + "  ".join(cells) + f"  {r['median_words']:5.0f}")
    print("\n* |Δ| > 2 SE. Primary cell: bp_v2_evidence_dev22, strict_all. Null control: bp_v2_rerun_dev22.")


if __name__ == "__main__":
    main()
