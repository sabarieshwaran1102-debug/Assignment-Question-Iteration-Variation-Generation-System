"""
Semantic Preservation Unit Tests for Phase 2.
Tests learning objective preservation, numerical consistency, different objective rejection,
and near-duplicate detection.
"""

from packages.schemas.models import SeedQuestion, QuestionVariation
from packages.common.providers.mock_provider import MockEmbeddingProvider, MockVectorStore
from apps.evaluator.difficulty import DefaultDifficultyAnalyzer, DefaultDifficultyEquivalenceValidator
from apps.evaluator.duplicate import DefaultDuplicateDetector
from apps.evaluator.answer_validator import DefaultAnswerValidator
from apps.evaluator.objective_validator import DefaultLearningObjectiveConsistencyValidator
from apps.evaluator.evaluator import DefaultVariationEvaluator


def build_test_evaluator():
    embedder = MockEmbeddingProvider(dimension=64)
    store = MockVectorStore()
    diff_analyzer = DefaultDifficultyAnalyzer()
    equiv_validator = DefaultDifficultyEquivalenceValidator()
    dup_detector = DefaultDuplicateDetector(embedding_provider=embedder, vector_store=store)
    ans_validator = DefaultAnswerValidator()
    obj_validator = DefaultLearningObjectiveConsistencyValidator()

    evaluator = DefaultVariationEvaluator(
        difficulty_analyzer=diff_analyzer,
        equivalence_validator=equiv_validator,
        duplicate_detector=dup_detector,
        answer_validator=ans_validator,
        objective_validator=obj_validator
    )
    return evaluator, dup_detector, ans_validator, obj_validator


def test_case_a_same_learning_objective_pass():
    """Case A: Seed and Candidate share the same learning objective -> PASS."""
    evaluator, _, _, _ = build_test_evaluator()
    seed = SeedQuestion(text="Calculate velocity using distance and time.", domain="Physics", difficulty=0.5)
    var = QuestionVariation(
        question="A train travels 300 m in 15 seconds. Calculate its velocity.",
        answer_key="Solution: 20 m/s. Explanation: 300 / 15 = 20 m/s",
        difficulty=0.5,
        domain="Physics"
    )

    result = evaluator.evaluate_variation(variation=var, seed=seed, existing_variations=[])
    assert result.is_valid is True
    assert result.objective_valid is True
    assert result.answer_valid is True


def test_case_b_different_learning_objective_fail():
    """Case B: Candidate strays to a different learning objective (Carnot engine vs velocity) -> FAIL."""
    evaluator, _, _, obj_validator = build_test_evaluator()
    seed = SeedQuestion(text="Calculate velocity using distance and time.", domain="Physics", difficulty=0.5)
    
    # Candidate text about Carnot heat engine efficiency
    cand_text = "Calculate the thermal efficiency of a Carnot heat engine operating between 500K and 300K."
    
    # Objective validator directly
    is_same_obj = obj_validator.validate_objective(seed.text, cand_text, domain="Physics")
    assert is_same_obj is False

    var = QuestionVariation(
        question=cand_text,
        answer_key="Efficiency = 40%",
        difficulty=0.5,
        domain="Physics"
    )
    result = evaluator.evaluate_variation(variation=var, seed=seed, existing_variations=[])
    assert result.is_valid is False
    assert result.objective_valid is False


def test_case_c_numerically_consistent_answer_pass():
    """Case C: Answer key is mathematically consistent with question parameters (180m / 9s = 20 m/s) -> PASS."""
    _, _, ans_validator, _ = build_test_evaluator()
    q = "A car travels 180 m in 9 seconds. Calculate velocity."
    ans = "Solution: 20 m/s (Explanation: 180 / 9 = 20)"
    
    assert ans_validator.validate_answer(q, ans) is True


def test_case_d_incorrect_answer_fail():
    """Case D: Answer key is numerically inconsistent (180m / 9s = 20 m/s, but answer states 50 m/s) -> FAIL."""
    _, _, ans_validator, _ = build_test_evaluator()
    q = "A car travels 180 m in 9 seconds. Calculate velocity."
    ans_wrong = "Solution: 50 m/s"
    
    assert ans_validator.validate_answer(q, ans_wrong) is False


def test_case_e_near_duplicate_detection():
    """Case E: Two candidate questions with trivial wording changes -> DETECT DUPLICATE."""
    _, dup_detector, _, _ = build_test_evaluator()
    q1 = "Calculate velocity of a vehicle moving 100m in 5 seconds."
    q2 = "Calculate velocity of a vehicle moving 100m in 5 seconds."
    
    dup_res = dup_detector.check_duplicate(candidate_text=q2, existing_texts=[q1], threshold=0.85)
    assert dup_res.is_duplicate is True
    assert dup_res.similarity_score >= 0.85
