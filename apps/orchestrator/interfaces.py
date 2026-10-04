"""
Interface for the Orchestrator module.
Coordinates end-to-end variation generation workflow from seed question parsing to evaluation.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from packages.schemas.models import GenerationRequest, GenerationResponse, GenerationMetrics, ReviewItem


class OrchestratorInterface(ABC):
    """Abstract interface for pipeline orchestrator."""

    @abstractmethod
    def orchestrate_generation(
        self, request: GenerationRequest
    ) -> Tuple[GenerationResponse, GenerationMetrics, List[ReviewItem]]:
        """Run complete generation pipeline for a seed question request."""
        pass
