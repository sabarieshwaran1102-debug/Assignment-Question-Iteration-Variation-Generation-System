"""
Bloom's Taxonomy Classifier and Evaluator module.
Classifies cognitive levels (Remember, Understand, Apply, Analyze, Evaluate, Create)
and evaluates Bloom equivalence to detect cognitive-level shifts.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import re

from packages.schemas.models import EvaluationResult

BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

BLOOM_KEYWORDS = {
    "Remember": [r"\b(define|list|recall|state|identify|name)\b"],
    "Understand": [r"\b(explain|describe|summarize|interpret|classify|discuss)\b"],
    "Apply": [r"\b(calculate|compute|solve|apply|determine|use|find)\b"],
    "Analyze": [r"\b(analyze|compare|contrast|differentiate|break down|examine)\b"],
    "Evaluate": [r"\b(evaluate|critique|judge|assess|justify|optimize)\b"],
    "Create": [r"\b(design|construct|formulate|devise|synthesize|compose)\b"],
}


class BloomClassifier(ABC):
    """Abstract interface for classifying Bloom's Taxonomy cognitive level."""

    @abstractmethod
    def classify_bloom(self, text: str, domain: Optional[str] = None) -> str:
        """Classify the cognitive Bloom level of a question text."""
        pass


class BloomEvaluator(ABC):
    """Abstract interface for evaluating Bloom cognitive equivalence."""

    @abstractmethod
    def evaluate_bloom_equivalence(self, seed_level: str, candidate_text: str, domain: Optional[str] = None) -> EvaluationResult:
        """Evaluate if candidate cognitive Bloom level is equivalent to seed level."""
        pass


class DefaultBloomClassifier(BloomClassifier):
    """Rule-based and semantic Bloom's Taxonomy Classifier."""

    def classify_bloom(self, text: str, domain: Optional[str] = None) -> str:
        text_lower = text.lower()
        for level in reversed(BLOOM_LEVELS):  # Check higher levels first
            patterns = BLOOM_KEYWORDS[level]
            for pat in patterns:
                if re.search(pat, text_lower):
                    return level
        return "Apply"  # Default for quantitative calculations


class DefaultBloomEvaluator(BloomEvaluator):
    """Evaluates Bloom taxonomy equivalence allowing configured neighbor tolerance."""

    def __init__(self, classifier: Optional[BloomClassifier] = None, max_level_distance: int = 1):
        self.classifier = classifier or DefaultBloomClassifier()
        self.max_level_distance = max_level_distance

    def evaluate_bloom_equivalence(self, seed_level: str, candidate_text: str, domain: Optional[str] = None) -> EvaluationResult:
        cand_level = self.classifier.classify_bloom(candidate_text, domain)

        seed_idx = BLOOM_LEVELS.index(seed_level) if seed_level in BLOOM_LEVELS else 2  # Apply=2
        cand_idx = BLOOM_LEVELS.index(cand_level) if cand_level in BLOOM_LEVELS else 2

        dist = abs(seed_idx - cand_idx)
        passed = dist <= self.max_level_distance

        confidence = 1.0 if dist == 0 else 0.85

        reasons = []
        if not passed:
            reasons.append(
                f"Bloom taxonomy shift detected: seed level is '{seed_level}' (idx {seed_idx}), "
                f"candidate level is '{cand_level}' (idx {cand_idx}), exceeding max level distance {self.max_level_distance}."
            )

        return EvaluationResult(
            passed=passed,
            confidence=confidence,
            score=1.0 - (dist * 0.2),
            reasons=reasons,
            metrics={
                "seed_bloom_level": seed_level,
                "candidate_bloom_level": cand_level,
                "level_distance": dist
            }
        )
