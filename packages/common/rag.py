"""
Local RAG / Knowledge Layer module.
Provides domain taxonomy, difficulty rubrics, Bloom taxonomy definitions,
and benchmark examples using local embedding vectors and vector storage.
Operates 100% locally with zero external API dependencies and fails gracefully if context is sparse.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from packages.schemas.models import RAGKnowledgeContext
from packages.common.providers.interfaces import EmbeddingProvider, VectorStore

BLOOM_DEFINITIONS = {
    "Remember": "Recall facts and basic concepts (define, list, state, identify).",
    "Understand": "Explain ideas or concepts (describe, summarize, interpret, classify).",
    "Apply": "Use information in new situations (calculate, compute, solve, apply).",
    "Analyze": "Draw connections among ideas (analyze, compare, contrast, differentiate).",
    "Evaluate": "Justify a stand or decision (evaluate, critique, judge, assess).",
    "Create": "Produce new or original work (design, construct, formulate, synthesize)."
}

DIFFICULTY_RUBRICS = {
    "Physics": {
        "easy": "Direct 1-step formula substitution (0.1 - 0.35)",
        "medium": "Standard 2-step dimensional calculation (0.36 - 0.65)",
        "hard": "Multi-variable vector optimization or non-uniform kinematics (0.66 - 1.0)"
    },
    "Computer Science": {
        "easy": "Direct array traversal or loop count (0.1 - 0.35)",
        "medium": "Standard tree lookup or recurrence relation (0.36 - 0.65)",
        "hard": "Lock-free concurrency or graph network flow (0.66 - 1.0)"
    }
}


class Retriever(ABC):
    """Abstract interface for knowledge retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge items for a query."""
        pass


class KnowledgeContextProvider(ABC):
    """Abstract interface for building RAG knowledge contexts."""

    @abstractmethod
    def get_context(self, domain: str, topic: Optional[str] = None, bloom_level: Optional[str] = None) -> RAGKnowledgeContext:
        """Provide structured RAG knowledge context for generation guidance."""
        pass


class DefaultRetriever(Retriever):
    """Local vector-store retriever."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            vec = self.embedding_provider.embed_text(query)
            results = self.vector_store.search(vec, top_k=top_k)
            return results
        except Exception:
            return []


class DefaultKnowledgeContextProvider(KnowledgeContextProvider):
    """Default local Knowledge Context Provider."""

    def __init__(self, retriever: Optional[Retriever] = None):
        self.retriever = retriever

    def get_context(self, domain: str, topic: Optional[str] = None, bloom_level: Optional[str] = None) -> RAGKnowledgeContext:
        rubric = DIFFICULTY_RUBRICS.get(domain, DIFFICULTY_RUBRICS["Physics"])
        b_level = bloom_level or "Apply"

        benchmark_examples = [
            "Calculate the velocity of a cyclist traveling 240m in 12s.",
            "Calculate the velocity of a train traveling 450m in 15s."
        ]

        patterns = [
            "Direct quantitative calculation (v = d / t)",
            "Real-world transportation scenario variation",
            "Target variable resolution with explicit units"
        ]

        return RAGKnowledgeContext(
            taxonomy={"domain": domain, "topic": topic or "General STEM"},
            topic_info=f"Core principles of {topic or domain} problem solving.",
            difficulty_rubric=rubric,
            bloom_definitions=BLOOM_DEFINITIONS,
            generation_patterns=patterns,
            benchmark_examples=benchmark_examples
        )
