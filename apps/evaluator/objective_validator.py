"""
Learning Objective Consistency Validator implementation.
Ensures candidate variations preserve the underlying pedagogical concept and subtopic of the seed question.
"""

import re
from typing import Optional, Set
from apps.evaluator.interfaces import LearningObjectiveValidator

SUBTOPIC_PATTERNS = {
    "kinematics": r"\b(velocity|speed|distance|displacement|acceleration|kinematics|motion|trajectory|projectile|stopping distance)\b",
    "thermodynamics": r"\b(carnot|heat engine|thermal efficiency|entropy|reservoir|thermodynamic|isothermal)\b",
    "circuits": r"\b(rlc|circuit|resistor|capacitor|inductor|voltage|current|impedance|resonance)\b",
    "cs_algorithms": r"\b(sorting|search|binary tree|asymptotic|dijkstra|graph|algorithm complexity)\b",
}


class DefaultLearningObjectiveConsistencyValidator(LearningObjectiveValidator):
    """Validates that candidate variation preserves the seed question learning objective."""

    def validate_objective(self, seed_question_text: str, candidate_text: str, domain: Optional[str] = None) -> bool:
        seed_lower = seed_question_text.lower()
        cand_lower = candidate_text.lower()

        seed_topics = self._get_subtopics(seed_lower)
        cand_topics = self._get_subtopics(cand_lower)

        # Cross-subtopic mismatch verification (e.g., Kinematics vs Thermodynamics)
        if "kinematics" in seed_topics and "thermodynamics" in cand_topics and "kinematics" not in cand_topics:
            return False

        if "thermodynamics" in seed_topics and "kinematics" in cand_topics and "thermodynamics" not in cand_topics:
            return False

        if "cs_algorithms" in seed_topics and ("thermodynamics" in cand_topics or "kinematics" in cand_topics):
            return False

        return True

    def _get_subtopics(self, text: str) -> Set[str]:
        found = set()
        for cat, pattern in SUBTOPIC_PATTERNS.items():
            if re.search(pattern, text):
                found.add(cat)
        return found
