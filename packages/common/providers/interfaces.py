"""
Clean provider interfaces for LLM, Embedding, and Vector Storage.
Decouples application code from specific provider implementations (Ollama, OpenAI, Gemini, Local models, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, TypeVar, Any
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract interface for LLM text and structured output generation."""

    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Generate unstructured text from a prompt."""
        pass

    def generate_text_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> List[str]:
        """Generate unstructured text for a batch of prompts.

        Providers can override this method to provide optimized batch inference.
        By default, it falls back to sequential calls to generate_text().
        """
        return [
            self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
            for prompt in prompts
        ]

    @abstractmethod
    def generate_structured(self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None, **kwargs: Any) -> T:
        """Generate structured Pydantic object from a prompt."""
        pass


class EmbeddingProvider(ABC):
    """Abstract interface for generating vector embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a single string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of strings."""
        pass


class VectorStore(ABC):
    """Abstract interface for vector indexing and similarity search."""

    @abstractmethod
    def add(self, item_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an item vector to the index."""
        pass

    @abstractmethod
    def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for top_k most similar vectors, returning items with similarity scores."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed items."""
        pass
