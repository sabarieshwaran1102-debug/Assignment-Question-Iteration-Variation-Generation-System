"""
Specialized Evaluation Layer module.
Provides independent, modular evaluators:
- ObjectiveEvaluator
- BloomEvaluator
- DifficultyEvaluator
- AnswerEvaluator
- DuplicateEvaluator
- QualityEvaluator
Combines evaluations into structured EvaluationResult instances without single opaque LLM calls.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from packages.schemas.models import (
    SeedQuestion,
    QuestionVariation,
    ValidationResult,
    DifficultyScore,
    EvaluationResult,
    DuplicateResult,
)
from apps.evaluator.interfaces import (
    LearningObjectiveValidator,
    AnswerValidator,
    DifficultyAnalyzer,
    DifficultyEquivalenceValidator,
    DuplicateDetector,
)
from apps.evaluator.bloom import BloomEvaluator, DefaultBloomEvaluator


class QualityEvaluator(ABC):
    """Abstract interface for master quality evaluation gate."""

    @abstractmethod
    def evaluate_candidate(
        self,
        candidate: QuestionVariation,
        seed: SeedQuestion,
        existing_variations: List[QuestionVariation],
        seed_bloom_level: str = "Apply"
    ) -> ValidationResult:
        """Evaluate candidate variation against all specialized quality gates."""
        pass


from apps.evaluator.blueprint_validator import DefaultBlueprintQuestionConsistencyValidator


class SpecializedQualityGate(QualityEvaluator):
    """Master Quality Gate coordinating specialized objective, bloom, difficulty, answer, duplicate, and blueprint consistency evaluators."""

    def __init__(
        self,
        objective_validator: LearningObjectiveValidator,
        answer_validator: AnswerValidator,
        difficulty_analyzer: DifficultyAnalyzer,
        equivalence_validator: DifficultyEquivalenceValidator,
        duplicate_detector: DuplicateDetector,
        bloom_evaluator: Optional[BloomEvaluator] = None,
        blueprint_validator: Optional[Any] = None
    ):
        self.objective_validator = objective_validator
        self.answer_validator = answer_validator
        self.difficulty_analyzer = difficulty_analyzer
        self.equivalence_validator = equivalence_validator
        self.duplicate_detector = duplicate_detector
        self.bloom_evaluator = bloom_evaluator or DefaultBloomEvaluator()
        self.blueprint_validator = blueprint_validator or DefaultBlueprintQuestionConsistencyValidator()

    def evaluate_candidate(
        self,
        candidate: QuestionVariation,
        seed: SeedQuestion,
        existing_variations: List[QuestionVariation],
        seed_bloom_level: str = "Apply"
    ) -> ValidationResult:
        reasons: List[str] = []

        # 1. Objective Validation
        obj_valid = self.objective_validator.validate_objective(
            seed_question_text=seed.text,
            candidate_text=candidate.question,
            domain=seed.domain
        )
        if not obj_valid:
            reasons.append("Learning objective mismatch: candidate alters core domain principle or subtopic.")

        # 2. Bloom Taxonomy Equivalence
        bloom_res = self.bloom_evaluator.evaluate_bloom_equivalence(
            seed_level=seed_bloom_level,
            candidate_text=candidate.question,
            domain=seed.domain
        )
        if not bloom_res.passed:
            reasons.extend(bloom_res.reasons)

        # 3. Answer Validation
        ans_valid = self.answer_validator.validate_answer(
            question_text=candidate.question,
            answer_key=candidate.answer_key
        )
        if not ans_valid:
            reasons.append("Answer key validation failure: answer is incomplete or mathematically inconsistent.")

        # 4. Blueprint & Question Consistency Validation
        bp_valid, bp_reasons = self.blueprint_validator.validate_blueprint_consistency(candidate)
        if not bp_valid:
            reasons.extend(bp_reasons)

        # 5. Difficulty Evaluation
        cand_diff_score = self.difficulty_analyzer.analyze_difficulty(
            question_text=candidate.question,
            domain=seed.domain
        )

        seed_diff_val = seed.difficulty if seed.difficulty is not None else 0.50
        seed_diff_score = DifficultyScore(score=seed_diff_val, blooms_level=seed_bloom_level)

        diff_equiv = self.equivalence_validator.validate_equivalence(
            seed_difficulty=seed_diff_score,
            variation_difficulty=cand_diff_score
        )
        if not diff_equiv:
            reasons.append(
                f"Difficulty shift flag: candidate score ({cand_diff_score.score:.2f}) "
                f"exceeds tolerance relative to seed ({seed_diff_score.score:.2f})."
            )

        # 6. Duplicate Check
        existing_texts = [v.question for v in existing_variations]
        dup_res = self.duplicate_detector.check_duplicate(
            candidate_text=candidate.question,
            existing_texts=existing_texts
        )
        candidate.duplicate_result = dup_res
        if dup_res.is_duplicate:
            reasons.append(f"Near-duplicate candidate: similarity score ({dup_res.similarity_score:.4f}) exceeds threshold.")

        is_valid = obj_valid and ans_valid and bp_valid and diff_equiv and not dup_res.is_duplicate and bloom_res.passed

        return ValidationResult(
            is_valid=is_valid,
            difficulty_score=cand_diff_score,
            is_difficulty_equivalent=diff_equiv,
            answer_valid=ans_valid,
            objective_valid=obj_valid,
            reasons=reasons
        )


# Class alias for Section 14 EvaluationCoordinator requirement
EvaluationCoordinator = SpecializedQualityGate

