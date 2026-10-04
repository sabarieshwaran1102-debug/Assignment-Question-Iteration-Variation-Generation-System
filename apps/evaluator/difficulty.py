"""
Default Difficulty Analyzer and Equivalence Validator implementations.
Computes readability, parameter complexity, and Bloom's level heuristics.
"""

import re
from typing import Optional
from packages.schemas.models import DifficultyScore
from apps.evaluator.interfaces import DifficultyAnalyzer, DifficultyEquivalenceValidator

BLOOMS_DIFFICULTY_MAP = {
    "Remember": 0.2,
    "Understand": 0.4,
    "Apply": 0.5,
    "Analyze": 0.7,
    "Evaluate": 0.85,
    "Create": 0.95
}


class DefaultDifficultyAnalyzer(DifficultyAnalyzer):
    """Analyzes text feature complexity to determine difficulty score."""

    def analyze_difficulty(self, question_text: str, domain: Optional[str] = None) -> DifficultyScore:
        words = re.findall(r"\w+", question_text)
        word_count = len(words)
        numbers = re.findall(r"\d+(?:\.\d+)?", question_text)
        num_count = len(numbers)
        avg_word_len = sum(len(w) for w in words) / max(1, word_count)
        
        # Determine Bloom's taxonomy level classification
        text_lower = question_text.lower()
        if any(w in text_lower for w in ["optimize", "evaluate", "derive", "critique"]):
            blooms = "Evaluate"
        elif any(w in text_lower for w in ["analyze", "compare", "contrast"]):
            blooms = "Analyze"
        elif any(w in text_lower for w in ["calculate", "compute", "solve", "determine"]):
            blooms = "Apply"
        else:
            blooms = "Understand"

        base_score = BLOOMS_DIFFICULTY_MAP.get(blooms, 0.5)
        # Minor adjustment for complexity (capped within +/- 0.05)
        adj = min(0.05, max(-0.05, (word_count - 15) * 0.002))
        normalized_score = round(min(1.0, max(0.1, base_score + adj)), 2)

        return DifficultyScore(
            score=normalized_score,
            blooms_level=blooms,
            complexity_factors={
                "word_count": word_count,
                "numerical_parameters": num_count,
                "avg_word_length": round(avg_word_len, 2),
                "domain": domain or "General"
            }
        )


class DefaultDifficultyEquivalenceValidator(DifficultyEquivalenceValidator):
    """Validates whether variation difficulty matches seed question baseline within tolerance."""

    def validate_equivalence(
        self,
        seed_difficulty: DifficultyScore,
        variation_difficulty: DifficultyScore,
        tolerance: float = 0.25
    ) -> bool:
        diff = abs(seed_difficulty.score - variation_difficulty.score)
        return diff <= tolerance
