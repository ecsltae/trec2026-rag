"""
Re-derive every figure the November 2026 paper revision adds or corrects, outside the official
score scripts (official_retrieval_scores.py, official_generation_scores.py).

    python eval/analysis/paper_revision_checks.py

1. dev22 nDCG@10 under the organisers' first (v1) and current (v2) qwen3.5-9b qrels.
2. The "no decomposition" baseline: 0.5268 was scored on a run that misses one topic; the same
   recipe on all 22 topics, and its top 10 against the submitted anchor-first ranking.
3. Pool size and number of search requests per topic, official119, both systems.
4. Relevant documents per narrative implied by the official per-narrative precision and recall.
5. The citation shift in systems/rag/generate.py _parse_answer when a citation follows the period.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from eval_rag import evaluate_run
from trec_shared.trec_utils import read_trec_run, load_qrels

V1 = ROOT / "data/rag/qrels/rag25-climbmix-umbrela-qwen3.5-9b.qrels"
V2 = ROOT / "data/rag/qrels_v2/rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels"
RUNS = {
    "anchor-first (bp_agentic_v2)": "bp_agentic_v2_dev22",
    "whole question, 22 topics": "api_bm25_reranked_full105",
    "whole question, 21 topics (EXPERIMENTS R3)": "api_bm25_reranked_eval22_v2",
    "plain BM25, full narrative": "api_climbmix_eval22_fullnarrative",
    "naive sub-questions": "agentic_iterative_sq_v1_full105",
    "probe-guided": "bp_agentic_v2_dev22_probe",
}


def section1_2():
    q1, q2 = load_qrels(str(V1)), load_qrels(str(V2))
    print("1-2. dev22 nDCG@10 / R@1000, v1 qrels and v2 qrels")
    for label, run in RUNS.items():
        r = read_trec_run(str(ROOT / "runs/rag" / f"{run}.tsv"))
        a, _ = evaluate_run(r, q1, cutoffs=(10, 1000))
        b, _ = evaluate_run(r, q2, cutoffs=(10, 1000))
        print(f"   {label:44s} topics {a['judged_topics']:2d}  v1 {a['ndcg@10']:.4f} / {a['r@1000']:.4f}"
              f"   v2 {b['ndcg@10']:.4f} / {b['r@1000']:.4f}")
    af = read_trec_run(str(ROOT / "runs/rag/bp_agentic_v2_dev22.tsv"))
    wq = read_trec_run(str(ROOT / "runs/rag/api_bm25_reranked_full105.tsv"))
    r3 = read_trec_run(str(ROOT / "runs/rag/api_bm25_reranked_eval22_v2.tsv"))
    same = sum([d for d, _, _ in af[t][:10]] == [d for d, _, _ in wq[t][:10]] for t in q1)
    print(f"   identical top 10, anchor-first vs whole question (22 topics): {same} of {len(q1)}")
    print(f"   topic missing from the 21-topic run: {sorted(set(q1) - set(r3))}")


def section3():
    print("3. official119 pool size and search requests per topic (narrative + anchor + facet queries)")
    for cache in ("official119", "official119_probe"):
        pools, searches = [], []
        for d in sorted((ROOT / "data/rag/cache/agentic_v2" / cache).iterdir()):
            if (d / "pool.jsonl").exists():
                pools.append(sum(1 for _ in open(d / "pool.jsonl")))
            if (d / "blueprint.json").exists():
                bp = json.loads((d / "blueprint.json").read_text())
                searches.append(2 + sum(len(f.get("queries", [])) for f in bp.get("facets", [])))
        print(f"   {cache:18s} pool median {statistics.median(pools):.0f} (range {min(pools)}-{max(pools)}, n={len(pools)})"
              f"   searches median {statistics.median(searches):.0f} (range {min(searches)}-{max(searches)})")


def section4():
    print("4. relevant documents per narrative implied by official P and R (primary, 82 narratives)")
    tsv = ROOT / "data/rag/official_scores/retrieval_2026-10-29/Bitem/retrieval-document-level-scores.tsv"
    rows = [r for r in csv.DictReader(open(tsv), delimiter="\t")
            if r["run"] == "bp_agentic_retrieval" and r["narrative-id"] != "all"]
    k = [int(r["num_ret"]) for r in rows]
    rel = [float(r["set_P"]) * int(r["num_ret"]) for r in rows]
    pool = [float(r["set_P"]) * int(r["num_ret"]) / float(r["set_recall"]) for r in rows if float(r["set_recall"]) > 0]
    ceil = [min(1.0, int(r["num_ret"]) / (float(r["set_P"]) * int(r["num_ret"]) / float(r["set_recall"])))
            for r in rows if float(r["set_recall"]) > 0]
    print(f"   depth median {statistics.median(k):.0f}; relevant returned median {statistics.median(rel):.0f};"
          f" relevant in judged pool median {statistics.median(pool):.0f} (n={len(pool)});"
          f" recall ceiling at our depth median {statistics.median(ceil):.2f}")


def section5():
    from systems.rag.generate import _parse_answer
    out = _parse_answer("Hospitals partner with schools. [shard_1] Mentors help. [shard_2]",
                        {"shard_1", "shard_2"}, 450, 30)
    print("5. _parse_answer on 'Hospitals partner with schools. [shard_1] Mentors help. [shard_2]':")
    print(f"   {out}")


if __name__ == "__main__":
    section1_2()
    section3()
    section4()
    section5()
