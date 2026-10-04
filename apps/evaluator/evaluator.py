"""
Default Variation Evaluator implementation.
Combines difficulty analysis, equivalence checking, answer validation,
duplicate detection, and learning objective consistency validation.
"""

from typing import List, Optional
from packages.schemas.models import SeedQuestion, QuestionVariation, ValidationResult
from apps.evaluator.interfaces import (
    VariationEvaluator,
    DifficultyAnalyzer,
    DifficultyEquivalenceValidator,
    DuplicateDetector,
    AnswerValidator,
    LearningObjectiveValidator,
)
from apps.evaluator.objective_validator import DefaultLearningObjectiveConsistencyValidator


class DefaultVariationEvaluator(VariationEvaluator):
    """Comprehensively evaluates generated question variations."""

    def __init__(
        self,
        difficulty_analyzer: DifficultyAnalyzer,
        equivalence_validator: DifficultyEquivalenceValidator,
        duplicate_detector: DuplicateDetector,
        answer_validator: AnswerValidator,
        objective_validator: Optional[LearningObjectiveValidator] = None,
        equivalence_tolerance: float = 0.25,
        duplicate_threshold: float = 0.85
    ):
        self.difficulty_analyzer = difficulty_analyzer
        self.equivalence_validator = equivalence_validator
        self.duplicate_detector = duplicate_detector
        self.answer_validator = answer_validator
        self.objective_validator = objective_validator or DefaultLearningObjectiveConsistencyValidator()
        self.equivalence_tolerance = equivalence_tolerance
        self.duplicate_threshold = duplicate_threshold

    def evaluate_variation(
        self,
        variation: QuestionVariation,
        seed: SeedQuestion,
        existing_variations: List[QuestionVariation]
    ) -> ValidationResult:
        reasons: List[str] = []

        # 1. Analyze Difficulty
        var_diff_score = self.difficulty_analyzer.analyze_difficulty(variation.question, domain=variation.domain)
        seed_diff_score = self.difficulty_analyzer.analyze_difficulty(seed.text, domain=seed.domain)
        
        if seed.difficulty is not None:
            seed_diff_score.score = seed.difficulty

        # 2. Check Difficulty Equivalence
        is_equivalent = self.equivalence_validator.validate_equivalence(
            seed_difficulty=seed_diff_score,
            variation_difficulty=var_diff_score,
            tolerance=self.equivalence_tolerance
        )
        if not is_equivalent:
            reasons.append(
                f"Difficulty discrepancy: seed={seed_diff_score.score:.2f}, variation={var_diff_score.score:.2f} (tolerance ±{self.equivalence_tolerance})"
            )

        # 3. Validate Answer Key & Numerical Consistency
        ans_valid = self.answer_validator.validate_answer(variation.question, variation.answer_key)
        if not ans_valid:
            reasons.append("Answer key is incomplete, missing, or mathematically inconsistent with question parameters.")

        # 4. Check Learning Objective Preservation
        obj_valid = self.objective_validator.validate_objective(seed.text, variation.question, domain=seed.domain)
        if not obj_valid:
            reasons.append(f"Learning objective mismatch: variation does not preserve core pedagogical concept of seed question.")

        # 5. Check Duplicates
        existing_texts = [v.question for v in existing_variations]
        dup_result = self.duplicate_detector.check_duplicate(
            candidate_text=variation.question,
            existing_texts=existing_texts,
            threshold=self.duplicate_threshold
        )
        if dup_result.is_duplicate:
            reasons.append(
                f"Duplicate detected: similarity score {dup_result.similarity_score:.2f} >= threshold {self.duplicate_threshold}"
            )

        is_valid = is_equivalent and ans_valid and obj_valid and (not dup_result.is_duplicate)

        return ValidationResult(
            is_valid=is_valid,
            difficulty_score=var_diff_score,
            is_difficulty_equivalent=is_equivalent,
            answer_valid=ans_valid,
            objective_valid=obj_valid,
            reasons=reasons
        )
