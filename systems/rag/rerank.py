"""
Cross-encoder reranker for TREC RAG 2026.

Reranks BM25/dense retrieved passages using a sentence-transformers cross-encoder.
GPU-accelerated when available.

Usage:
    from systems.rag.rerank import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, candidates, top_k=100)
    # candidates: list of (seg_id, text, score)
    # returns: list of (seg_id, score) sorted by cross-encoder score
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Wraps sentence-transformers CrossEncoder for passage reranking.

    The ms-marco-MiniLM-L-6-v2 model is fast (~5ms/pair on GPU) and strong
    for passage retrieval tasks. For higher quality at cost of speed, use
    cross-encoder/ms-marco-electra-base.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None,
                 batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("sentence-transformers required: pip install sentence-transformers") from e

        device = self._device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading cross-encoder %s on %s", self.model_name, device)
        self._model = CrossEncoder(self.model_name, device=device, max_length=512)
        logger.info("Cross-encoder loaded.")

    def rerank(self, query: str, candidates: list, top_k: int = 100) -> list:
        """
        Rerank candidates with cross-encoder.

        Args:
            query: query string
            candidates: list of (seg_id, text, initial_score) or (seg_id, text)
            top_k: how many to return

        Returns:
            list of (seg_id, ce_score) sorted descending
        """
        if not candidates:
            return []
        self._load()

        pairs = []
        seg_ids = []
        for item in candidates:
            seg_id = item[0]
            text = item[1]
            seg_ids.append(seg_id)
            # Truncate text to avoid exceeding max_length
            pairs.append((query, text[:800]))

        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        ranked = sorted(zip(seg_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def rerank_with_fusion(self, query: str, candidates: list, top_k: int = 100,
                           alpha: float = 0.5) -> list:
        """
        Linear interpolation of initial score and cross-encoder score.
        Useful when initial BM25/dense scores carry signal.

        Args:
            alpha: weight on CE score (1-alpha on initial score)
            candidates: must be (seg_id, text, initial_score) triples
        """
        if not candidates:
            return []
        self._load()

        # Normalize initial scores to [0,1]
        init_scores = [float(c[2]) for c in candidates]
        max_s = max(init_scores) or 1.0
        min_s = min(init_scores) or 0.0
        rng = max_s - min_s or 1.0
        norm_init = [(s - min_s) / rng for s in init_scores]

        pairs = [(query, c[1][:800]) for c in candidates]
        seg_ids = [c[0] for c in candidates]
        ce_scores_raw = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        # Sigmoid-normalize CE scores (they're logits for ms-marco models)
        import math
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))

        fused = []
        for seg_id, ce_raw, n_init in zip(seg_ids, ce_scores_raw, norm_init):
            ce_norm = sigmoid(ce_raw)
            fused.append((seg_id, alpha * ce_norm + (1 - alpha) * n_init))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]
