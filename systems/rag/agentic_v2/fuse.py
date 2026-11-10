"""
Stage 4 — Fuse: weighted RRF of anchor + facet rankings, with slot guarantee.

Anchor weight 2.0 protects top-10 precision (it IS the validated 0.527 recipe);
facet lists act on recall and coverage. Slot guarantee swaps each facet's top
docs into the tail of the final top-k so every facet is represented in the
R TSV.
"""
import json
from pathlib import Path


def weighted_rrf(rankings: dict, weights: dict, k: int = 60) -> list:
    """Weighted reciprocal rank fusion.

    Args:
        rankings: {list_name: [[docid, score], ...]}
        weights: {list_name: float}
    Returns:
        [[docid, fused_score], ...] descending
    """
    fused: dict = {}
    for name, ranking in rankings.items():
        w = weights.get(name, 1.0)
        for rank, (docid, _score) in enumerate(ranking, 1):
            fused[docid] = fused.get(docid, 0.0) + w / (k + rank)
    return sorted(([d, s] for d, s in fused.items()),
                  key=lambda x: x[1], reverse=True)


def fuse_topic(d: Path, anchor_weight: float = 2.0, slot_guarantee: int = 2,
               top_k: int = 100, rrf_k: int = 60, depth: int = 1000,
               strategy: str = "anchor_first"):
    """Produce final_ranking.json for one topic dir.

    strategy="anchor_first" (default — Gate 1 winner, 2026-07: NDCG@10 0.5352
    vs 0.5268 baseline, R@1000 0.254 vs 0.222):
        head = anchor CE-fusion list verbatim (precision),
        then facet-RRF fill (coverage), then pool-BM25 tail to depth (recall).
    strategy="rrf": weighted RRF of all lists + slot guarantee (kept for ablations;
        scored 0.449 — facet consensus dilutes the anchor head).
    """
    rankings = json.loads((d / "facet_rankings.json").read_text())
    if not rankings:
        (d / "final_ranking.json").write_text(json.dumps([]))
        return

    if strategy == "anchor_first":
        seen = set()
        ordered = []
        for docid, _ in rankings.get("_anchor", []):
            if docid not in seen:
                seen.add(docid)
                ordered.append(docid)
        facet_only = {n: r for n, r in rankings.items() if n != "_anchor"}
        if facet_only:
            for docid, _ in weighted_rrf(facet_only,
                                         {n: 1.0 for n in facet_only}, k=rrf_k):
                if docid not in seen:
                    seen.add(docid)
                    ordered.append(docid)
        final = [[docid, float(depth - i)] for i, docid in enumerate(ordered)]
    else:  # "rrf"
        weights = {name: (anchor_weight if name == "_anchor" else 1.0)
                   for name in rankings}
        fused = weighted_rrf(rankings, weights, k=rrf_k)
        final = fused[:top_k]
        if slot_guarantee > 0:
            in_final = {docid for docid, _ in final}
            must_have = []
            for name, ranking in rankings.items():
                if name == "_anchor":
                    continue
                for docid, _score in ranking[:slot_guarantee]:
                    if docid not in in_final and docid not in {m for m, _ in must_have}:
                        must_have.append((docid, name))
            for i, (docid, _facet) in enumerate(must_have):
                pos = len(final) - 1 - i
                if pos < top_k // 2:
                    break  # never let guarantees eat into the top half
                final[pos] = [docid, final[pos][1]]

    # Tail fill to depth from the pool (BM25 order), skipping head docs
    if depth > len(final):
        pool_path = d / "pool.jsonl"
        if pool_path.exists():
            from systems.rag.agentic_v2.harvest import read_pool
            in_final = {docid for docid, _ in final}
            floor = final[-1][1] if final else 1.0
            tail_rank = 0
            for e in read_pool(pool_path):
                if len(final) >= depth:
                    break
                if e["docid"] in in_final:
                    continue
                tail_rank += 1
                final.append([e["docid"], floor * 0.9 - tail_rank * 1e-6])

    # Re-monotonize scores so rank order and score order agree
    for i in range(1, len(final)):
        if final[i][1] >= final[i - 1][1]:
            final[i][1] = final[i - 1][1] * 0.999

    (d / "final_ranking.json").write_text(json.dumps(final))
