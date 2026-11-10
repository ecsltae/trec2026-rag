"""
Stage 3 — Tighten: per-topic local dense retrieval over the harvested pool.

Embeds the pool with a sentence-transformers model (A100), builds an in-memory
FAISS FlatIP index (CPU — trivial at pool size ~2.5k), then per facet:
dense top-N per facet query -> union -> cross-encoder rerank.
The anchor list is the proven 0.527 recipe applied to the wider pool:
dense top-300 by narrative -> CE rerank with the full narrative.

VRAM note: this phase assumes Ollama models are UNLOADED (pipeline handles it).
Embeddings are persisted (embeddings.npy + emb_docids.json) so reruns are free.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from systems.rag.agentic_v2.harvest import read_pool


class TightenStage:
    def __init__(self, embed_model="BAAI/bge-large-en-v1.5",
                 rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 dense_per_query=150, anchor_dense_k=300,
                 facet_keep=100, anchor_keep=200):
        self.embed_model_name = embed_model
        self.rerank_model_name = rerank_model
        self.dense_per_query = dense_per_query
        self.anchor_dense_k = anchor_dense_k
        self.facet_keep = facet_keep
        self.anchor_keep = anchor_keep
        self._embedder = None
        self._reranker = None

    def _load_models(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            self._embedder = SentenceTransformer(self.embed_model_name, device="cuda")
            self._reranker = CrossEncoder(self.rerank_model_name, device="cuda",
                                          max_length=512)
            print(f"  [tighten] loaded {self.embed_model_name} + "
                  f"{self.rerank_model_name}", flush=True)

    # -- embeddings ---------------------------------------------------------

    def _embed_pool(self, d: Path, pool: list) -> tuple:
        """Return (embeddings [n,dim] normalized, docids). Cached on disk."""
        emb_path = d / "embeddings.npy"
        ids_path = d / "emb_docids.json"
        docids = [e["docid"] for e in pool]
        if emb_path.exists() and ids_path.exists():
            cached_ids = json.loads(ids_path.read_text())
            if cached_ids == docids:
                return np.load(emb_path), docids
        self._load_models()
        texts = [e["text"][:2000] for e in pool]
        emb = self._embedder.encode(texts, batch_size=128, show_progress_bar=False,
                                    normalize_embeddings=True,
                                    convert_to_numpy=True).astype(np.float32)
        np.save(emb_path, emb)
        ids_path.write_text(json.dumps(docids))
        return emb, docids

    def _embed_queries(self, queries: list) -> np.ndarray:
        self._load_models()
        # bge models want a retrieval instruction prefix on queries
        prefixed = [f"Represent this sentence for searching relevant passages: {q}"
                    for q in queries]
        return self._embedder.encode(prefixed, normalize_embeddings=True,
                                     convert_to_numpy=True).astype(np.float32)

    # -- per-topic ----------------------------------------------------------

    def _load_fusion_reranker(self):
        """The validated 0.527 reranker (CE+initial-score fusion, text[:800])."""
        if getattr(self, "_fusion_reranker", None) is None:
            from systems.rag.rerank import CrossEncoderReranker
            self._fusion_reranker = CrossEncoderReranker(
                model_name=self.rerank_model_name)
            self._fusion_reranker._load()
        return self._fusion_reranker

    def run_topic(self, d: Path, narrative: str,
                  out_name: str = "facet_rankings.json"):
        """Produce facet_rankings.json (or a named variant for ablations).

        Anchor list replicates the validated 0.527 recipe EXACTLY, applied to the
        pool: top-200 by BM25 -> rerank_with_fusion (0.5*CE + 0.5*norm BM25).
        (Ablation 2026-07: dense-by-narrative candidates + raw CE scored 0.264 —
        never regress to that.)
        Facet lists: dense per facet query -> rerank_with_fusion with dense
        similarity as the initial score.
        """
        import faiss

        pool = read_pool(d / "pool.jsonl")
        if not pool:
            (d / out_name).write_text(json.dumps({}))
            return
        bp = json.loads((d / "blueprint.json").read_text())
        emb, docids = self._embed_pool(d, pool)
        text_by_id = {e["docid"]: e["text"] for e in pool}
        reranker = self._load_fusion_reranker()

        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        rankings = {}

        # Facet lists: dense per facet query -> rerank_with_fusion(dense sim)
        for facet in bp["facets"]:
            qs = facet["queries"]
            q_emb = self._embed_queries(qs)
            k = min(self.dense_per_query, len(pool))
            sims, idx = index.search(q_emb, k)
            best_sim = {}
            for row_s, row_i in zip(sims, idx):
                for s, j in zip(row_s, row_i):
                    if j >= 0:
                        did = docids[j]
                        best_sim[did] = max(best_sim.get(did, -1.0), float(s))
            candidates = [(did, text_by_id[did], sim)
                          for did, sim in best_sim.items()]
            joined_query = " ; ".join(qs)
            scored = reranker.rerank_with_fusion(joined_query, candidates,
                                                 top_k=self.facet_keep)
            rankings[facet["id"]] = [[did, float(s)] for did, s in scored]

        # Anchor list: the 0.527 recipe on the pool — BM25 top-200 -> CE fusion
        bm25_top = sorted(pool, key=lambda e: e["bm25_score"], reverse=True)[:200]
        candidates = [(e["docid"], e["text"], e["bm25_score"]) for e in bm25_top]
        scored = reranker.rerank_with_fusion(narrative, candidates,
                                             top_k=self.anchor_keep)
        rankings["_anchor"] = [[did, float(s)] for did, s in scored]

        (d / out_name).write_text(json.dumps(rankings))
