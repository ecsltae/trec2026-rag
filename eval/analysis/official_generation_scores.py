#!/usr/bin/env python3
"""Analyse the official TREC 2026 RAG generation scores (organisers' email of 2026-11-04).

Reads the seven TSVs the organisers sent and prints:

  1. overall scores against the field: human side-by-side Bradley-Terry ratings and battle counts
     (our two priority runs, two assessor batches), rubric and nugget scores (all four runs)
  2. standings: per narrative against the field median, with an exact sign test
  3. our four runs head to head, per narrative
  4. our two selection instruments against the official scores: the human annotator's blind
     votes (bp_manual_curated) and our citation-support judge (bp_support_argmax), each judged by
     whether it picked the officially better of v2 and probe on a narrative
  5. repeatability of the official rubric judge: the composites copy v2 or probe verbatim, so the
     same answer was scored twice
  6. answer length against the official scores

Batch 1: 17 narratives, assessors wrote rubrics from intuition, then compared answers side by side.
Batch 2: 25 narratives, assessors nuggetized corpus shards, then compared side by side.
Rubric and nugget assignment is automatic (GPT-6.1 Sol). Only v2 and probe were side-by-side
assessed ("top-2 priority runs"). The size of the generation field is not stated.

Usage:  python eval/analysis/official_generation_scores.py
"""
import csv
import json
import sqlite3
from collections import Counter
from math import comb
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OFF = ROOT / "data/rag/official_scores/generation_2026-11-03/Bitem"
CURATION = ROOT / "data/rag/curation/curation.db"
V2, PROBE, MANUAL, ARGMAX = "bp_agentic_v2", "bp_probe_guided", "bp_manual_curated", "bp_support_argmax"
RUNS = [V2, PROBE, MANUAL, ARGMAX]
SCORED = [("batch 1, human-intuition rubric", "rag-rubric-scores-batch1-human.tsv"),
          ("batch 2, NIST nuggets", "rag-nugget-scores-batch2-manual.tsv"),
          ("batch 2, automatic nuggets", "rag-nugget-scores-batch2-automatic.tsv")]
METRICS = ["vital_strict", "all_strict"]


def rows(name):
    return list(csv.DictReader(open(OFF / name), delimiter="\t"))


def sign_test(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def per_topic(rr, run, m):
    return {r["narrative-id"]: float(r[m]) for r in rr if r["run"] == run and r["narrative-id"] != "all"}


def answers(run):
    out = {}
    for line in open(ROOT / f"submissions/{run}/rag_output_trec_rag_2026.jsonl"):
        r = json.loads(line)
        out[r["metadata"]["narrative_id"]] = [s["text"] for s in r["answer"]]
    return out


def picks(composite, v2, probe):
    """Which source run a composite copied on each narrative: exact match first, else the source
    sharing more sentences (two manual answers were edited after composition, by the G21 gate)."""
    out = {}
    for t, sents in composite.items():
        a, b = len(set(sents) & set(v2[t])), len(set(sents) & set(probe[t]))
        out[t] = "same" if v2[t] == probe[t] else (V2 if a > b else PROBE if b > a else "?")
    return out


def main():
    scored = {label: rows(f) for label, f in SCORED}

    print("1. Overall (narrative-id = all)")
    for b in (1, 2):
        bt = {r["run"]: r for r in rows(f"rag-bt-scores-batch{b}.tsv")}
        sxs = {r["run"]: r for r in rows(f"rag-sxs-counts-batch{b}.tsv") if r["narrative-id"] == "all"}
        any_r = next(iter(bt.values()))
        print(f"   side by side, batch {b}: field BT min {any_r['bt_score_min']}, median {any_r['bt_score_median']},"
              f" max {any_r['bt_score_max']}")
        for run in (V2, PROBE):
            r, s = bt[run], sxs[run]
            w, t, l = int(s["wins"]), int(s["ties"]), int(s["losses"])
            below = float(r["ci_upper"]) < float(r["bt_score_median"])
            print(f"      {run:18s} BT {r['bt_score']:>7s} [{r['ci_lower']}, {r['ci_upper']}]"
                  f"  {'95% CI wholly below the median' if below else 'CI reaches the median'}"
                  f"   battles {w}-{t}-{l} (W-T-L), wins {w / (w + l):.1%} of decided")
    for label, rr in scored.items():
        overall = {r["run"]: r for r in rr if r["narrative-id"] == "all"}
        any_r = overall[V2]
        print(f"   {label}: field median vital {any_r['vital_strict_median']} all {any_r['all_strict_median']}"
              f"  | max vital {any_r['vital_strict_max']} all {any_r['all_strict_max']}")
        for run in RUNS:
            r = overall[run]
            half = " ".join("above" if float(r[m]) > float(r[m + "_median"]) else
                            "at" if float(r[m]) == float(r[m + "_median"]) else "below" for m in METRICS)
            print(f"      {run:18s} vital {r['vital_strict']}  all {r['all_strict']}   vs median: {half}")

    print("\n2. Standings per narrative against the field median (above / tie / below, exact sign test)")
    for label, rr in scored.items():
        for m in METRICS:
            for run in RUNS:
                x = [r for r in rr if r["run"] == run and r["narrative-id"] != "all"]
                above = sum(float(r[m]) > float(r[m + "_median"]) for r in x)
                below = sum(float(r[m]) < float(r[m + "_median"]) for r in x)
                print(f"   {label:32s} {m:12s} {run:18s} {above:2d} / {len(x) - above - below:2d} / {below:2d}"
                      f"   p = {sign_test(above, below):.2g}")
    for b in (1, 2):
        for run in (V2, PROBE):
            x = [r for r in rows(f"rag-sxs-counts-batch{b}.tsv") if r["run"] == run and r["narrative-id"] != "all"]
            more = sum(int(r["wins"]) > int(r["losses"]) for r in x)
            fewer = sum(int(r["wins"]) < int(r["losses"]) for r in x)
            print(f"   side by side batch {b}  {run:18s} narratives won more battles than lost: {more},"
                  f" fewer: {fewer}, even: {len(x) - more - fewer}")

    print("\n3. Head to head per narrative (first wins / ties / second wins, exact sign test)")
    for a, b in [(V2, PROBE), (MANUAL, V2), (MANUAL, PROBE), (ARGMAX, V2), (ARGMAX, PROBE), (MANUAL, ARGMAX)]:
        for label, rr in scored.items():
            m = "all_strict"
            x, y = per_topic(rr, a, m), per_topic(rr, b, m)
            w, l = sum(x[t] > y[t] for t in x), sum(x[t] < y[t] for t in x)
            print(f"   {a:18s} vs {b:18s} {label:32s} {m}: {w:2d} / {len(x) - w - l:2d} / {l:2d}   p = {sign_test(w, l):.2g}")

    v2a, pra = answers(V2), answers(PROBE)
    votes = {t: v for t, v in sqlite3.connect(CURATION).execute(
        "select topic_id, human_verdict from battles where set_name = 'official119_v2_vs_probe'")}
    votes = {t: {"bp_agentic_v2": V2, "probe_guided": PROBE}.get(v, "skip") for t, v in votes.items()}
    instruments = {"human votes (bp_manual_curated)": (MANUAL, votes),
                   "support judge (bp_support_argmax)": (ARGMAX, picks(answers(ARGMAX), v2a, pra))}

    print("\n4. Our selection instruments against the official scores: on narratives where v2 and probe"
          " score differently, did the instrument pick the better one?")
    for name, (composite, pick) in instruments.items():
        print(f"   {name}: picks {dict(Counter(pick.values()))}")
        for label, rr in scored.items():
            for m in METRICS:
                v, p = per_topic(rr, V2, m), per_topic(rr, PROBE, m)
                right = sum(1 for t in v if pick.get(t) in (V2, PROBE) and v[t] != p[t]
                            and (v[t] > p[t]) == (pick[t] == V2))
                wrong = sum(1 for t in v if pick.get(t) in (V2, PROBE) and v[t] != p[t]
                            and (v[t] > p[t]) != (pick[t] == V2))
                chosen = [v[t] if pick.get(t) == V2 else p[t] if pick.get(t) == PROBE else (v[t] + p[t]) / 2 for t in v]
                rand = sum((v[t] + p[t]) / 2 for t in v) / len(v)
                best = sum(max(v[t], p[t]) for t in v) / len(v)
                print(f"      {label:32s} {m:12s} right {right:2d}, wrong {wrong:2d} (p = {sign_test(right, wrong):.2g})"
                      f"   picked {sum(chosen) / len(chosen):.4f} vs random {rand:.4f} vs best {best:.4f}")
        # Pooled over the two human-written item sets, whose narratives are disjoint.
        right = wrong = 0
        for label in ("batch 1, human-intuition rubric", "batch 2, NIST nuggets"):
            v, p = per_topic(scored[label], V2, "all_strict"), per_topic(scored[label], PROBE, "all_strict")
            for t in v:
                if pick.get(t) in (V2, PROBE) and v[t] != p[t]:
                    right += (v[t] > p[t]) == (pick[t] == V2)
                    wrong += (v[t] > p[t]) != (pick[t] == V2)
        print(f"      human-written items pooled, all_strict   right {right:2d}, wrong {wrong:2d}"
              f" (p = {sign_test(right, wrong):.2g})")
        # The official human battles do not pit v2 against probe directly, and the files do not name
        # the opponents. The target is the share of its decided battles a run won on that narrative.
        for b in (1, 2):
            share = {}
            for r in rows(f"rag-sxs-counts-batch{b}.tsv"):
                w, l = int(r["wins"]), int(r["losses"])
                if r["narrative-id"] != "all" and w + l:
                    share[(r["run"], r["narrative-id"])] = w / (w + l)
            ts = [t for (run, t) in share if run == V2 and (PROBE, t) in share and share[(V2, t)] != share[(PROBE, t)]
                  and pick.get(t) in (V2, PROBE)]
            right = sum((share[(V2, t)] > share[(PROBE, t)]) == (pick[t] == V2) for t in ts)
            print(f"      side by side batch {b}, share of battles won  right {right:2d}, wrong {len(ts) - right:2d}"
                  f" (p = {sign_test(right, len(ts) - right):.2g})")

    print("\n5. Repeatability of the official rubric judge: composite answers identical to a source answer")
    rescored = changed = 0
    for name, (composite, _) in instruments.items():
        comp = answers(composite)
        same = {t: (V2 if comp[t] == v2a[t] else PROBE) for t in comp if comp[t] in (v2a[t], pra[t])}
        for rr in scored.values():
            score = {(r["run"], r["narrative-id"]): tuple(r[m] for m in METRICS) for r in rr}
            for t in same:
                if (composite, t) in score:
                    rescored += 1
                    changed += score[(composite, t)] != score[(same[t], t)]
        for label, rr in scored.items():
            for m in METRICS:
                c, src = per_topic(rr, composite, m), {run: per_topic(rr, run, m) for run in (V2, PROBE)}
                pairs = [(c[t], src[same[t]][t]) for t in c if t in same]
                diff = [abs(x - y) for x, y in pairs]
                print(f"   {composite:18s} {label:32s} {m:12s} identical answers {len(pairs):2d}:"
                      f" same score {sum(d == 0 for d in diff):2d}, mean |diff| {sum(diff) / len(diff):.3f},"
                      f" max {max(diff):.3f}")
    print(f"   in all: {rescored} identical answers scored twice, {changed} with a different score")

    print("\n6. Answer length (words) against the official scores")
    lengths = {run: {t: len(" ".join(s).split()) for t, s in answers(run).items()} for run in (V2, PROBE)}
    for run in (V2, PROBE):
        w = sorted(lengths[run].values())
        print(f"   {run:18s} words per answer: mean {sum(w) / len(w):.0f}, median {w[len(w) // 2]},"
              f" range {w[0]}-{w[-1]} (limit 1024)")
    for label, rr in scored.items():
        xs, ys = [], []
        for run in (V2, PROBE):
            s = per_topic(rr, run, "all_strict")
            xs += [lengths[run][t] for t in s]
            ys += list(s.values())
        rho, p = spearmanr(xs, ys)
        v, pr = per_topic(rr, V2, "all_strict"), per_topic(rr, PROBE, "all_strict")
        longer_wins = sum(1 for t in v if v[t] != pr[t] and (v[t] > pr[t]) == (lengths[V2][t] > lengths[PROBE][t]))
        decided = sum(1 for t in v if v[t] != pr[t] and lengths[V2][t] != lengths[PROBE][t])
        print(f"   {label:32s} Spearman(words, all_strict) {rho:+.2f} (p={p:.2g}, n={len(xs)});"
              f" the longer of v2/probe scores higher on {longer_wins} of {decided}")
    for b in (1, 2):
        xs, ys = [], []
        for run in (V2, PROBE):
            for r in rows(f"rag-sxs-counts-batch{b}.tsv"):
                if r["run"] == run and r["narrative-id"] != "all" and int(r["wins"]) + int(r["losses"]):
                    xs.append(lengths[run][r["narrative-id"]])
                    ys.append(int(r["wins"]) / (int(r["wins"]) + int(r["losses"])))
        rho, p = spearmanr(xs, ys)
        print(f"   side by side batch {b}: Spearman(words, share of decided battles won) {rho:+.2f} (p={p:.2g}, n={len(xs)})")


if __name__ == "__main__":
    main()
