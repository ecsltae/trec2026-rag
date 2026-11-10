"""
Table 1 of the paper, re-derived with the share of each top ten that the development labels judged.

Audit of 2026-11-10. The development labels were pooled from whole-question retrieval, so a list
that promotes facet documents has unjudged documents in its top ten, and nDCG@10 counts those as
non-relevant. For every row of Table 1 that can be rebuilt, this prints nDCG@10 under the v1 and v2
labels, the judged share of the top ten (J@10), nDCG@10 over judged documents only (the condensed
list, an upper bound for the same row), and the paired difference from anchor-first both ways.

    source ~/TREC/venv/bin/activate
    python eval/analysis/fusion_judged_share.py        # CPU, about a minute; writes the .json next to it

Rows:
  anchor-first              data/rag/cache/agentic_v2/dev22/<topic>/final_ranking.json
  whole question only       runs/rag/api_bm25_reranked_full105.tsv
  RRF, whole question x2/x8 rebuilt from the cached rankings with fuse.py's weighted_rrf and slot
                            guarantee (the run files were not kept; the rebuild reproduces the logged
                            v1 figures, 0.449 and 0.527 in EXPERIMENTS.md R5)
  cross-encoder alone       the pool's BM25 top 200 ordered by the raw MiniLM score against the
                            narrative, i.e. the whole-question recipe without the BM25 term
  naive sub-questions       runs/rag/agentic_iterative_sq_v1_full105.tsv
  plain BM25                runs/rag/api_climbmix_eval22_fullnarrative.tsv
  anchor-first, bge         head of the cached facet_rankings_bge.json whole-question list (R7)
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from eval_rag import ndcg_at_k
from trec_shared.trec_utils import load_qrels, read_trec_run
from systems.rag.agentic_v2.fuse import weighted_rrf
from systems.rag.agentic_v2.harvest import read_pool
from systems.rag.agentic_v2.pipeline import load_topic_set

CACHE = ROOT / "data/rag/cache/agentic_v2/dev22"
Q1 = load_qrels(str(ROOT / "data/rag/qrels/rag25-climbmix-umbrela-qwen3.5-9b.qrels"))
Q2 = load_qrels(str(ROOT / "data/rag/qrels_v2/rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels"))
T = sorted(Q1)
OUT = Path(__file__).with_suffix(".json")


def rrf(t, w, top_k=100, slot=2):
    """fuse.py, strategy="rrf": weighted RRF of all rankings plus the slot guarantee."""
    rk = json.loads((CACHE / t / "facet_rankings.json").read_text())
    final = weighted_rrf(rk, {n: (w if n == "_anchor" else 1.0) for n in rk}, k=60)[:top_k]
    in_final, must = {d for d, _ in final}, []
    for n, r in rk.items():
        if n == "_anchor":
            continue
        for d, _ in r[:slot]:
            if d not in in_final and d not in {m for m, _ in must}:
                must.append((d, n))
    for i, (d, _) in enumerate(must):
        pos = len(final) - 1 - i
        if pos < top_k // 2:
            break
        final[pos] = [d, final[pos][1]]
    return [d for d, _ in final]


def cross_encoder_alone():
    import torch
    torch.set_num_threads(4)          # the GPU and most cores may be busy with a writer job
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512)
    topics = load_topic_set("dev22")
    out = {}
    for t in T:
        pool = sorted(read_pool(CACHE / t / "pool.jsonl"), key=lambda e: e["bm25_score"], reverse=True)[:200]
        s = ce.predict([(topics[t].get("narrative", ""), e["text"][:800]) for e in pool], batch_size=64,
                       show_progress_bar=False)
        out[t] = [e["docid"] for e, _ in sorted(zip(pool, s), key=lambda x: -x[1])]
    return out


def run(name):
    r = read_trec_run(str(ROOT / "runs/rag" / f"{name}.tsv"))
    return {t: [d for d, _, _ in r[t]] for t in T}


def main():
    lists = {
        "anchor-first": {t: [d for d, _ in json.loads((CACHE / t / "final_ranking.json").read_text())] for t in T},
        "whole question only": run("api_bm25_reranked_full105"),
        "RRF, whole question x2": {t: rrf(t, 2) for t in T},
        "RRF, whole question x8": {t: rrf(t, 8) for t in T},
        "cross-encoder alone": cross_encoder_alone(),
        "naive sub-questions": run("agentic_iterative_sq_v1_full105"),
        "plain BM25": run("api_climbmix_eval22_fullnarrative"),
        "anchor-first, bge reranker": {t: [d for d, _ in json.loads(
            (CACHE / t / "facet_rankings_bge.json").read_text())["_anchor"]] for t in T},
    }
    mean = lambda xs: sum(xs) / len(xs)
    cond = lambda l, q: [d for d in l if d in q]
    per = {n: {"v1": {t: ndcg_at_k(L[t], Q1[t], 10) for t in T},
               "v2": {t: ndcg_at_k(L[t], Q2[t], 10) for t in T},
               "v1c": {t: ndcg_at_k(cond(L[t], Q1[t]), Q1[t], 10) for t in T},
               "v2c": {t: ndcg_at_k(cond(L[t], Q2[t]), Q2[t], 10) for t in T},
               "j": {t: sum(d in Q2[t] for d in L[t][:10]) / 10 for t in T}} for n, L in lists.items()}
    res = {}
    print(f"{'':28s} {'v1':>7s} {'v2':>7s} {'J@10':>6s} {'v1 judged':>10s} {'v2 judged':>10s}"
          f"   anchor-first minus row, v2: plain (SE) | judged only (SE)")
    base = per["anchor-first"]
    for n, p in per.items():
        row = {k: mean(p[k].values()) for k in p}
        d1 = [base["v2"][t] - p["v2"][t] for t in T]
        d2 = [base["v2c"][t] - p["v2c"][t] for t in T]
        se = lambda d: st.stdev(d) / len(d) ** 0.5
        row.update({"diff_plain": mean(d1), "se_plain": se(d1), "diff_judged": mean(d2), "se_judged": se(d2),
                    "judged_record": [sum(x > 0 for x in d2), sum(x < 0 for x in d2)]})
        res[n] = row
        print(f"{n:28s} {row['v1']:7.4f} {row['v2']:7.4f} {row['j']:6.3f} {row['v1c']:10.4f} {row['v2c']:10.4f}"
              f"   {row['diff_plain']:+.4f} ({row['se_plain']:.4f}) | {row['diff_judged']:+.4f} ({row['se_judged']:.4f})"
              f"  {row['judged_record'][0]}-{row['judged_record'][1]}")
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
