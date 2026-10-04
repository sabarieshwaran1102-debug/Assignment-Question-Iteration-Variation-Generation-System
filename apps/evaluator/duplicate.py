"""
Default Duplicate Detector implementation.
Leverages vector embeddings and similarity indexing to detect duplicate and near-duplicate question variations.
"""

from typing import List
import numpy as np
from packages.schemas.models import DuplicateResult
from packages.common.providers.interfaces import EmbeddingProvider, VectorStore
from apps.evaluator.interfaces import DuplicateDetector


class DefaultDuplicateDetector(DuplicateDetector):
    """Detects duplicates using embedding similarity search."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def check_duplicate(
        self,
        candidate_text: str,
        existing_texts: List[str],
        threshold: float = 0.85
    ) -> DuplicateResult:
        if not existing_texts:
            return DuplicateResult(
                is_duplicate=False,
                similarity_score=0.0,
                matched_question_id=None
            )

        candidate_vec = np.array(self.embedding_provider.embed_text(candidate_text), dtype=np.float64)
        norm_cand = np.linalg.norm(candidate_vec)
        if norm_cand > 0:
            candidate_vec = candidate_vec / norm_cand

        max_sim = 0.0
        matched_id = None

        for idx, text in enumerate(existing_texts):
            existing_vec = np.array(self.embedding_provider.embed_text(text), dtype=np.float64)
            norm_exist = np.linalg.norm(existing_vec)
            if norm_exist > 0:
                existing_vec = existing_vec / norm_exist
                sim = float(np.dot(candidate_vec, existing_vec))
                sim = max(0.0, min(1.0, sim))
                if sim > max_sim:
                    max_sim = sim
                    matched_id = f"var_{idx+1}"

        is_dup = max_sim >= threshold
        return DuplicateResult(
            is_duplicate=is_dup,
            similarity_score=round(max_sim, 4),
            matched_question_id=matched_id if is_dup else None
        )
