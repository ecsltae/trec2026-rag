"""
Retrieval module for TREC RAG 2026.

Classes:
    ApiRetriever   — Pyserini REST API (primary for TREC RAG 2026, ClimbMix-400b)
    BM25Retriever  — local rank_bm25 index (MS MARCO V2.1, dev/fallback)
    DenseRetriever — local FAISS FlatIP index + sentence-transformers (dev/fallback)
    HybridRetriever — RRF fusion of BM25 + Dense

All retrievers expose: .retrieve(query, top_k) -> list of (seg_id, score)
ApiRetriever additionally caches doc texts: .get_text(seg_id) -> str
"""
import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.trec_utils import reciprocal_rank_fusion


class ApiRetriever:
    """Pyserini REST API retriever for TREC RAG 2026.

    Primary retriever — points at ClimbMix-400b (or msmarco-v2.1-doc-segmented).
    Caches doc texts during retrieve() so the pipeline avoids extra fetches.

    Token setup (official):
        echo "PYSERINI_API_TOKEN=your_token_here" > .env.local
        (request token by emailing get-pyserini@googlegroups.com)
    """

    BASE_URL = "http://api.castorini.uwaterloo.ca"

    def __init__(self, index="climbmix-400b", token=None, timeout=30):
        self.index = index
        self.timeout = timeout
        self._doc_cache: dict = {}
        # Load token: arg > PYSERINI_API_TOKEN env (official) > PYSERINI_TOKEN env (legacy) > .env.local
        self._token = (token
                       or os.environ.get("PYSERINI_API_TOKEN")
                       or os.environ.get("PYSERINI_TOKEN")
                       or self._load_token())
        if not self._token:
            raise RuntimeError(
                "Pyserini API token not found. "
                "Set PYSERINI_API_TOKEN in .env.local or as env var. "
                "Request token: email get-pyserini@googlegroups.com"
            )

    def _load_token(self):
        env_file = Path(__file__).parent.parent.parent / ".env.local"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("PYSERINI_API_TOKEN=") or line.startswith("PYSERINI_TOKEN="):
                    return line.split("=", 1)[1].strip()
        return None

    @property
    def _headers(self):
        return {"Authorization": f"Bearer {self._token}"}

    def health_check(self):
        r = requests.get(f"{self.BASE_URL}/", headers=self._headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def retrieve(self, query, top_k=100):
        """Search the index. Returns list of (seg_id, score), caches doc texts."""
        import time as _time
        import re
        import json as _json
        import unicodedata

        # Normalize query: NFKC normalization + manual replacements
        # NFKC converts smart quotes, ligatures, and other compatibility chars to standard ASCII
        query = unicodedata.normalize('NFKC', query)
        # Additional manual replacements for chars that NFKC might miss
        query = query.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        query = query.replace('—', '-').replace('–', '-')

        url = f"{self.BASE_URL}/v1/{self.index}/search"
        params = {"query": query, "hits": top_k}

        # Retry loop: handles 429 rate-limit and transient errors
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, headers=self._headers, timeout=self.timeout)
            except requests.exceptions.RequestException as e:
                if attempt == 3:
                    print(f"  WARNING: API connection failed for '{query[:60]}': {e}", flush=True)
                    return []
                _time.sleep(15 * (attempt + 1))
                continue

            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30 * (attempt + 1)))
                print(f"  429 rate-limit on attempt {attempt+1} — waiting {wait}s...", flush=True)
                _time.sleep(wait)
                continue

            r.raise_for_status()
            break
        else:
            return []

        try:
            data = r.json()
        except requests.exceptions.JSONDecodeError:
            text = r.text
            # Fix invalid escape sequences while preserving valid JSON escapes
            text = re.sub(r'\\([^"\\/bfnrtux])', lambda m: '\\\\' + m.group(1), text)
            text = re.sub(r'\\x(?![0-9a-fA-F]{2})', r'\\\\x', text)
            try:
                data = _json.loads(text, strict=False)
            except Exception as e2:
                print(f"  WARNING: unparseable API response for '{query[:60]}' ({e2})", flush=True)
                # Fallback: retry with first sentence (more semantically meaningful than char truncation)
                first_sent = query.split('.')[0] + '.' if '.' in query else query[:150]
                if first_sent != query and len(first_sent) < len(query):
                    print(f"  Retrying with first sentence ({len(first_sent)} chars)...", flush=True)
                    return self.retrieve(first_sent, top_k=top_k)
                return []
        results = []
        for candidate in data.get("candidates", []):
            docid = candidate["docid"]
            score = float(candidate["score"])
            # Cache the doc text to avoid re-fetching during generation
            doc = candidate.get("doc", "")
            text = doc if isinstance(doc, str) else (doc.get("text") or doc.get("segment") or doc.get("body") or "")
            self._doc_cache[docid] = text
            results.append((docid, score))
        return results

    def get_text(self, seg_id):
        """Return cached text, or fetch from API if not cached."""
        if seg_id in self._doc_cache:
            return self._doc_cache[seg_id]
        url = f"{self.BASE_URL}/v1/{self.index}/doc/{seg_id}"
        try:
            r = requests.get(url, headers=self._headers, timeout=self.timeout)
            if r.status_code == 404:
                return ""
            r.raise_for_status()
            doc = r.json().get("doc", {})
            # Handle both dict and string responses
            if isinstance(doc, str):
                text = doc
            else:
                text = doc.get("text") or doc.get("segment") or doc.get("body") or ""
            self._doc_cache[seg_id] = text
            return text
        except (requests.exceptions.JSONDecodeError, requests.exceptions.RequestException) as e:
            # ClimbMix API sometimes returns malformed JSON (multi-MB docs with invalid control chars)
            # Return empty string to gracefully skip these segments
            self._doc_cache[seg_id] = ""
            return ""


class BM25Retriever:
    def __init__(self, index_path):
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.index = data["index"]
        self.seg_ids = data["seg_ids"]

    def retrieve(self, query, top_k=100):
        tokens = query.lower().split()
        scores = self.index.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.seg_ids[i], float(scores[i])) for i in top_idx]


class DenseRetriever:
    def __init__(self, index_path, meta_path=None, model_name=None):
        self.index = faiss.read_index(str(index_path))
        meta_path = meta_path or Path(index_path).with_suffix(".meta")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.seg_ids = meta["seg_ids"]
        model = model_name or meta.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)

    def retrieve(self, query, top_k=100):
        emb = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.seg_ids[idx], float(score)))
        return results


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k=60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def retrieve(self, query, bm25_k=200, dense_k=200, final_k=100):
        bm25_results = self.bm25.retrieve(query, top_k=bm25_k)
        dense_results = self.dense.retrieve(query, top_k=dense_k)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=self.rrf_k)
        return fused[:final_k]


def build_retriever(config):
    """Factory: build a retriever from a config dict (from config.yaml)."""
    method = config["retrieval"]["method"]
    idx_dir = Path(config["paths"].get("index_dir", "data/rag/indexes"))

    ret_cfg = config["retrieval"]
    bm25_fname = ret_cfg.get("bm25_index", "bm25.pkl")
    dense_fname = ret_cfg.get("dense_index", "dense.faiss")

    if method == "api":
        api_cfg = ret_cfg.get("api", {})
        return ApiRetriever(
            index=api_cfg.get("index", "climbmix-400b"),
            token=api_cfg.get("token"),
            timeout=api_cfg.get("timeout", 30),
        )
    elif method == "bm25":
        return BM25Retriever(idx_dir / bm25_fname)
    elif method == "dense":
        return DenseRetriever(idx_dir / dense_fname)
    elif method == "hybrid":
        bm25 = BM25Retriever(idx_dir / bm25_fname)
        dense = DenseRetriever(idx_dir / dense_fname)
        return HybridRetriever(bm25, dense, rrf_k=ret_cfg["rrf_k"])
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
