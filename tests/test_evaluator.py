"""
Unit tests for Evaluator module components.
"""

from packages.schemas.models import SeedQuestion, QuestionVariation, DifficultyScore
from packages.common.providers.mock_provider import MockEmbeddingProvider, MockVectorStore
from apps.evaluator.difficulty import DefaultDifficultyAnalyzer, DefaultDifficultyEquivalenceValidator
from apps.evaluator.duplicate import DefaultDuplicateDetector
from apps.evaluator.answer_validator import DefaultAnswerValidator
from apps.evaluator.evaluator import DefaultVariationEvaluator


def test_difficulty_analyzer():
    analyzer = DefaultDifficultyAnalyzer()
    diff = analyzer.analyze_difficulty("Calculate velocity when distance is 100m and time is 5s", domain="Physics")
    assert 0.0 <= diff.score <= 1.0
    assert diff.blooms_level != ""
    assert "word_count" in diff.complexity_factors


def test_difficulty_equivalence_validator():
    validator = DefaultDifficultyEquivalenceValidator()
    s1 = DifficultyScore(score=0.5)
    s2 = DifficultyScore(score=0.6)
    s3 = DifficultyScore(score=0.9)

    assert validator.validate_equivalence(s1, s2, tolerance=0.25) is True
    assert validator.validate_equivalence(s1, s3, tolerance=0.25) is False


def test_duplicate_detector():
    embedder = MockEmbeddingProvider(dimension=64)
    store = MockVectorStore()
    detector = DefaultDuplicateDetector(embedding_provider=embedder, vector_store=store)

    q1 = "Calculate velocity of car moving 100m in 5s"
    q2 = "Calculate velocity of vehicle moving 100m in 5s"
    q3 = "Explain bubble sort time complexity in computer science"

    v1 = embedder.embed_text(q1)
    store.add("id1", v1, {"text": q1})

    res_dup = detector.check_duplicate(candidate_text=q2, existing_texts=[q1], threshold=0.70)
    assert res_dup.similarity_score > 0.6

    res_diff = detector.check_duplicate(candidate_text=q3, existing_texts=[q1], threshold=0.85)
    assert res_diff.is_duplicate is False


def test_answer_validator():
    validator = DefaultAnswerValidator()
    assert validator.validate_answer("What is 2+2?", "4 (Explanation: 2+2=4)") is True
    assert validator.validate_answer("What is 2+2?", "") is False
    assert validator.validate_answer("What is 2+2?", "   ") is False


def test_variation_evaluator():
    embedder = MockEmbeddingProvider(dimension=64)
    store = MockVectorStore()
    diff_analyzer = DefaultDifficultyAnalyzer()
    equiv_validator = DefaultDifficultyEquivalenceValidator()
    dup_detector = DefaultDuplicateDetector(embedding_provider=embedder, vector_store=store)
    ans_validator = DefaultAnswerValidator()

    evaluator = DefaultVariationEvaluator(
        difficulty_analyzer=diff_analyzer,
        equivalence_validator=equiv_validator,
        duplicate_detector=dup_detector,
        answer_validator=ans_validator
    )

    seed = SeedQuestion(text="Calculate velocity of car moving 100m in 5s", domain="Physics", difficulty=0.5)
    var = QuestionVariation(
        question="In a physics scenario, compute speed for 120m in 6s",
        answer_key="Speed = 20 m/s. Step 1: Divide distance by time.",
        difficulty=0.5,
        domain="Physics"
    )

    result = evaluator.evaluate_variation(variation=var, seed=seed, existing_variations=[])
    assert result.is_valid is True
    assert result.answer_valid is True
    assert result.is_difficulty_equivalent is True
