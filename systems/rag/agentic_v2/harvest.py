"""
Stage 2 — Harvest: go wide via the Pyserini API.

Multi-query BM25 pooling: full narrative + anchor query + every facet query.
Persists docids, texts, provenance and BM25 scores to pool.jsonl so no API
call is ever repeated (the doc text cache in ApiRetriever is memory-only).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from systems.rag.retrieve import ApiRetriever


def harvest_topic(retriever: ApiRetriever, blueprint: dict, narrative: str,
                  hits_narrative: int = 500, hits_anchor: int = 300,
                  hits_facet: int = 200) -> list:
    """Run all blueprint searches for one topic, pool unique docs.

    Returns list of pool entries:
        {"docid", "text", "bm25_score" (max over queries), "sources": [query labels]}
    """
    searches = [("narrative", narrative[:2000], hits_narrative),
                ("anchor", blueprint["anchor_query"], hits_anchor)]
    for facet in blueprint["facets"]:
        for qi, q in enumerate(facet["queries"]):
            searches.append((f"{facet['id']}.q{qi}", q, hits_facet))

    pool: dict = {}  # docid -> entry
    for label, query, hits in searches:
        results = retriever.retrieve(query, top_k=hits)
        t0 = time.time()
        for docid, score in results:
            if docid in pool:
                entry = pool[docid]
                entry["bm25_score"] = max(entry["bm25_score"], score)
                entry["sources"].append(label)
            else:
                pool[docid] = {
                    "docid": docid,
                    "text": retriever.get_text(docid) or "",
                    "bm25_score": score,
                    "sources": [label],
                }
        print(f"    [{label}] '{query[:50]}...' -> {len(results)} hits "
              f"(pool now {len(pool)})", flush=True)

    # Drop empty-text docs (can't embed or cite them)
    entries = [e for e in pool.values() if e["text"].strip()]
    entries.sort(key=lambda e: e["bm25_score"], reverse=True)
    return entries


def write_pool(path: Path, entries: list):
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def read_pool(path: Path) -> list:
    return [json.loads(line) for line in open(path) if line.strip()]
