"""
Coverage that counts only grounded sentences (addendum B of PREREG_post_results_rerun.md).

The official generation measures credit an item when the answer TEXT supports it; nothing checks
that the sentence's citation supports it too. This builds, for each dev22 run that has per-sentence
support judgments, two filtered copies of the answers:

  _grounded   keeps a sentence only if at least one of its citations got FULL support
  _supported  keeps a sentence only if at least one of its citations got full or PARTIAL support

The filtered runs are then scored on the official dev nuggets with the same local judge as the
originals (eval/nugget_local_judge.py), so the difference between a run and its filtered copies is
the coverage that rests on sentences our support judge found unbacked by what they cite.

    python eval/analysis/grounded_coverage_dev22.py build     # writes runs/rag/<run>_{grounded,supported}.jsonl
    python eval/analysis/grounded_coverage_dev22.py report    # after nugget_local_judge has scored them

Support judgments: runs/rag/<run>.support_judgments.jsonl (eval/ragdoll_local_judge.py, qwen3:32b,
RAGDoll support prompt, 4,000-character truncation). Same sentence order as the run file; checked.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "runs/rag"
BASES = ["bp_agentic_v2_dev22", "bp_v2_coverage_dev22", "bp_v2_evidence_dev22", "bp_v2_multipass_dev22"]
VARIANTS = {"grounded": lambda sup: any(s == 2 for s in sup),
            "supported": lambda sup: any(s >= 1 for s in sup)}
SUMMARY = ROOT / "eval/analysis/nugget_dev22_grounded.json"


def build():
    for base in BASES:
        judged = {r["topic_id"]: r for r in map(json.loads, open(RUNS / f"{base}.support_judgments.jsonl"))}
        rows = [json.loads(l) for l in open(RUNS / f"{base}.jsonl")]
        for name, keep in VARIANTS.items():
            kept_s = total_s = kept_w = total_w = 0
            with open(RUNS / f"{base}_{name}.jsonl", "w") as f:
                for r in rows:
                    tid = str(r["metadata"]["topic_id"])
                    js = judged[tid]["sentences"]
                    assert len(js) == len(r["answer"]), f"{base} {tid}: sentence count differs"
                    out = []
                    for s, j in zip(r["answer"], js):
                        assert s["text"] == j["text"], f"{base} {tid}: sentence text differs"
                        w = len(s["text"].split())
                        total_s += 1; total_w += w
                        if keep([c["support"] for c in j["citations"]]):
                            out.append(s); kept_s += 1; kept_w += w
                    f.write(json.dumps({**r, "answer": out}) + "\n")
            print(f"{base}_{name}: kept {kept_s}/{total_s} sentences ({kept_s / total_s:.0%}), "
                  f"{kept_w}/{total_w} words")


def paired(a, b):
    d = [b[t] - a[t] for t in sorted(set(a) & set(b))]
    se = st.stdev(d) / len(d) ** 0.5
    return st.mean(d), se


def report():
    runs = {}
    for f in ("nugget_dev22_existing.json", "nugget_dev22_arms.json", "nugget_dev22_multipass.json",
              "nugget_dev22_grounded.json", "nugget_dev22_review.json"):
        if not (ROOT / "eval/analysis" / f).exists():
            continue
        runs.update(json.load(open(ROOT / "eval/analysis" / f))["runs"])
    pt = {n: {t: v["strict_all_score"] for t, v in r["per_topic"].items()} for n, r in runs.items()}
    print(f"{'run':30s} {'as written':>10s} {'supported':>10s} {'grounded':>10s}   lost to ungrounded sentences")
    for base in BASES:
        a, s, g = (runs[base]["strict_all_score"], runs[f"{base}_supported"]["strict_all_score"],
                   runs[f"{base}_grounded"]["strict_all_score"])
        dm, se = paired(pt[base], pt[f"{base}_grounded"])
        print(f"{base:30s} {a:10.4f} {s:10.4f} {g:10.4f}   {dm:+.4f} (SE {se:.4f})")
    for v in ("", "_supported", "_grounded"):
        dm, se = paired(pt[f"bp_agentic_v2_dev22{v}"], pt[f"bp_v2_evidence_dev22{v}"])
        print(f"evidence - shipped{v or ' (as written)':>22s}: {dm:+.4f}  SE {se:.4f}  {dm / se:+.1f} SE")


if __name__ == "__main__":
    {"build": build, "report": report}[sys.argv[1]]()
