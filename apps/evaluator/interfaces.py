"""
Interfaces for the Evaluator module.
Defines contracts for difficulty analysis, equivalence checking, duplicate detection,
answer key validation, learning objective validation, and overall variation quality evaluation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from packages.schemas.models import (
    SeedQuestion,
    QuestionVariation,
    DifficultyScore,
    ValidationResult,
    DuplicateResult,
    AnswerKey,
)


class DifficultyAnalyzer(ABC):
    """Abstract interface for analyzing question text difficulty."""

    @abstractmethod
    def analyze_difficulty(self, question_text: str, domain: Optional[str] = None) -> DifficultyScore:
        """Calculate normalized difficulty score (0.0 to 1.0) and complexity factors."""
        pass


class DifficultyEquivalenceValidator(ABC):
    """Abstract interface for checking difficulty equivalence between seed and variation."""

    @abstractmethod
    def validate_equivalence(
        self,
        seed_difficulty: DifficultyScore,
        variation_difficulty: DifficultyScore,
        tolerance: float = 0.25
    ) -> bool:
        """Return True if variation difficulty is within tolerance of seed difficulty."""
        pass


class DuplicateDetector(ABC):
    """Abstract interface for detecting duplicates and near-duplicates."""

    @abstractmethod
    def check_duplicate(
        self,
        candidate_text: str,
        existing_texts: List[str],
        threshold: float = 0.85
    ) -> DuplicateResult:
        """Check candidate variation against existing variations for semantic similarity."""
        pass


class AnswerValidator(ABC):
    """Abstract interface for validating variation answer key quality and numerical consistency."""

    @abstractmethod
    def validate_answer(self, question_text: str, answer_key: str) -> bool:
        """Validate whether the answer key is non-empty, correct, and mathematically consistent."""
        pass


class LearningObjectiveValidator(ABC):
    """Abstract interface for validating learning objective preservation."""

    @abstractmethod
    def validate_objective(self, seed_question_text: str, candidate_text: str, domain: Optional[str] = None) -> bool:
        """Return True if candidate question preserves seed question's pedagogical learning objective."""
        pass


class VariationEvaluator(ABC):
    """Abstract interface for evaluating overall variation quality."""

    @abstractmethod
    def evaluate_variation(
        self,
        variation: QuestionVariation,
        seed: SeedQuestion,
        existing_variations: List[QuestionVariation]
    ) -> ValidationResult:
        """Comprehensively evaluate a candidate variation against seed and dataset."""
        pass
