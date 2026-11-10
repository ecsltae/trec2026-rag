#!/usr/bin/env python3
"""Analyse the official TREC 2026 RAG retrieval scores (organisers' email of 2026-11-02).

Reads the three TSVs the organisers sent (document-level UMBRELA set_P / set_recall over 82
narratives; nugget gain over 25 narratives, NIST and automatic nuggets) and prints:

  1. overall scores against the field (min / median / max over 114 runs, 27 teams)
  2. per narrative: how often each run beats the field median
  3. primary vs probe head to head, with an exact sign test
  4. our own relevance labeller (UMBRELA prompts on qwen3:32b) against the official judge,
     on precision: coverage, absolute level, per-topic rank agreement, calibration line
  5. whether deeper topics cost precision
  6. saturation of NIST-nugget gain
  7. standings: which half of the 114 runs each run is in, and its per-narrative record against
     the median run (exact sign test); the talk deck's standings slide shows the same figures

The official qrels are not distributed yet, so (4) compares per-topic set precision recomputed
from our labels over exactly the documents we submitted, at exactly the submitted depth.

Usage:  python eval/analysis/official_retrieval_scores.py
"""
import csv
from collections import defaultdict
from math import comb
from pathlib import Path

from scipy.stats import linregress, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OFF = ROOT / "data/rag/official_scores/retrieval_2026-10-29/Bitem"
OUR_QRELS = ROOT / "data/rag/qrels/official119-umbrela-qwen3-32b-bing.qrels"
RUNS = {
    "bp_agentic_retrieval": ROOT / "submissions/bp_agentic_v2/r_output_trec_rag_2026.tsv",
    "bp_probe_ret": ROOT / "submissions/bp_probe_guided/r_output_trec_rag_2026.tsv",
}
PRIMARY, PROBE = "bp_agentic_retrieval", "bp_probe_ret"


def rows(name):
    return list(csv.DictReader(open(OFF / name), delimiter="\t"))


def sign_test(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    doc = rows("retrieval-document-level-scores.tsv")
    nug = {"NIST nuggets": rows("retrieval-nugget-level-scores-manual.tsv"),
           "auto nuggets": rows("retrieval-nugget-level-scores-automatic.tsv")}

    print("1. Overall (narrative-id = all), field = 114 runs / 27 teams")
    for r in doc:
        if r["narrative-id"] == "all":
            print(f"   {r['run']:22s} set_P {r['set_P']}  set_recall {r['set_recall']}  depth {r['num_ret']}"
                  f"   | field median P {r['set_P_median']}, R {r['set_recall_median']}, depth {r['num_ret_median']}")
    for label, rr in nug.items():
        for r in rr:
            if r["narrative-id"] == "all":
                print(f"   {r['run']:22s} {label} gain {r['gain']}  effort {r['effort']}"
                      f"   | field min {r['gain_min']} median {r['gain_median']} max {r['gain_max']}")

    print("\n2. Per narrative against the field median (above / tie / below; at field max)")
    for label, rr, m in [("set_P", doc, "set_P"), ("set_recall", doc, "set_recall"),
                         ("NIST gain", nug["NIST nuggets"], "gain"), ("auto gain", nug["auto nuggets"], "gain")]:
        for run in (PRIMARY, PROBE):
            x = [r for r in rr if r["run"] == run and r["narrative-id"] != "all"]
            above = sum(float(r[m]) > float(r[m + "_median"]) for r in x)
            tie = sum(float(r[m]) == float(r[m + "_median"]) for r in x)
            atmax = sum(float(r[m]) >= float(r[m + "_max"]) for r in x)
            print(f"   {label:10s} {run:22s} {above:2d} / {tie:2d} / {len(x) - above - tie:2d}   at max {atmax:2d}   (n={len(x)})")

    print("\n3. Primary vs probe, per narrative (primary wins / ties / losses, exact sign test)")
    for label, rr, m in [("set_P", doc, "set_P"), ("set_recall", doc, "set_recall"),
                         ("NIST gain", nug["NIST nuggets"], "gain"), ("auto gain", nug["auto nuggets"], "gain")]:
        a = {r["narrative-id"]: float(r[m]) for r in rr if r["run"] == PRIMARY and r["narrative-id"] != "all"}
        p = {r["narrative-id"]: float(r[m]) for r in rr if r["run"] == PROBE and r["narrative-id"] != "all"}
        w, l = sum(a[k] > p[k] for k in a), sum(a[k] < p[k] for k in a)
        print(f"   {label:10s} {w:2d} / {len(a) - w - l:2d} / {l:2d}   p = {sign_test(w, l):.4f}")

    ours = {}
    for line in open(OUR_QRELS):
        t, _, d, g = line.split()
        ours[(t, d)] = int(g)
    official = defaultdict(dict)
    for r in doc:
        if r["narrative-id"] != "all":
            official[r["run"]][r["narrative-id"]] = (float(r["set_P"]), int(r["num_ret"]))

    print("\n4. Our labeller (UMBRELA prompts, qwen3:32b) against the official judge, on set precision")
    for run, path in RUNS.items():
        docs = defaultdict(list)
        for line in open(path):
            t, _, d, *_ = line.rstrip("\n").split("\t")
            docs[t].append(d)
        topics = sorted(official[run])
        covered = sum((t, d) in ours for t in topics for d in docs[t])
        total = sum(len(docs[t]) for t in topics)
        depth_ok = sum(len(docs[t]) == official[run][t][1] for t in topics)
        p_ours = [sum(ours.get((t, d), 0) >= 2 for d in docs[t]) / len(docs[t]) for t in topics]
        p_off = [official[run][t][0] for t in topics]
        mo, mf = sum(p_ours) / len(p_ours), sum(p_off) / len(p_off)
        rho, prho = spearmanr(p_ours, p_off)
        r, pr = pearsonr(p_ours, p_off)
        fit = linregress(p_ours, p_off)
        # Pooled counts give a rigorous bound: |ours - official| >= |ours| - |official|.
        our_pos = sum(sum(ours.get((t, d), 0) >= 2 for d in docs[t]) for t in topics)
        off_pos = sum(round(official[run][t][0] * official[run][t][1]) for t in topics)
        print(f"   {run}: coverage {covered}/{total} = {covered / total:.1%}, depth matches {depth_ok}/{len(topics)}")
        print(f"      mean set_P ours {mo:.4f} vs official {mf:.4f}  ({mo / mf:.2f}x as lenient)")
        print(f"      pooled: we call {our_pos} of {total} relevant, the official judge {off_pos};"
              f" at least {(our_pos - off_pos) / our_pos:.0%} of our 'relevant' calls are officially non-relevant")
        print(f"      per-topic Spearman {rho:.3f} (p={prho:.1e}), Pearson {r:.3f} (p={pr:.1e})")
        print(f"      calibration: official = {fit.intercept:.3f} + {fit.slope:.3f} x ours  (R^2 {fit.rvalue ** 2:.2f})")

        depth = [official[run][t][1] for t in topics]
        rho_d, p_d = spearmanr(depth, p_off)
        srt = sorted(zip(depth, p_off))
        n = len(srt)
        terciles = [srt[: n // 3], srt[n // 3: 2 * n // 3], srt[2 * n // 3:]]
        desc = " | ".join(f"depth {g[0][0]}-{g[-1][0]}: P {sum(x[1] for x in g) / len(g):.3f}" for g in terciles)
        print(f"   5. depth vs official precision: Spearman {rho_d:.3f} (p={p_d:.1e})   {desc}")

    x = [r for r in nug["NIST nuggets"] if r["run"] == PRIMARY and r["narrative-id"] != "all"]
    print(f"\n6. NIST-nugget gain saturation over {len(x)} topics: field max = 1.0 on"
          f" {sum(float(r['gain_max']) >= 0.9999 for r in x)}, field median = 1.0 on"
          f" {sum(float(r['gain_median']) >= 0.9999 for r in x)}, primary = 1.0 on"
          f" {sum(float(r['gain']) >= 0.9999 for r in x)}")

    # The organisers publish min / median / max per metric, not ranks. With 114 runs the median
    # averages the 57th and 58th values, so scoring above it means rank <= 57, below it rank >= 58.
    print("\n7. Standings: which half of the 114 runs, and per narrative against the median run")
    print(f"   {'metric':24s} {'run':22s} {'score':>7s} {'median':>7s}  half                per-narrative above/below  sign p")
    for label, rr, m in [("set precision", doc, "set_P"), ("set recall", doc, "set_recall"),
                         ("gain, automatic nuggets", nug["auto nuggets"], "gain"),
                         ("gain, NIST nuggets", nug["NIST nuggets"], "gain")]:
        overall = {r["run"]: r for r in rr if r["narrative-id"] == "all"}
        for run in (PRIMARY, PROBE):
            score, med = float(overall[run][m]), float(overall[run][m + "_median"])
            half = "top (rank <= 57)" if score > med else ("bottom (rank >= 58)" if score < med else "at median")
            per = [r for r in rr if r["run"] == run and r["narrative-id"] != "all"]
            above = sum(float(r[m]) > float(r[m + "_median"]) for r in per)
            below = sum(float(r[m]) < float(r[m + "_median"]) for r in per)
            print(f"   {label:24s} {run:22s} {score:7.4f} {med:7.4f}  {half:19s} {above:3d} / {below:<3d} (ties {len(per) - above - below:2d})"
                  f"      {sign_test(above, below):.2g}")


if __name__ == "__main__":
    main()
