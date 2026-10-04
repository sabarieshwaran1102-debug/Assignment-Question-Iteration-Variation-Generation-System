"""
Unit tests for shared Pydantic and domain schemas.
"""

import pytest
from pydantic import ValidationError

from packages.schemas.models import (
    SeedQuestion,
    GenerationRequest,
    VariationOutput,
    GenerationResponse,
    QuestionVariation,
    DifficultyScore,
    ValidationResult,
    DuplicateResult,
    AnswerKey,
    ReviewItem,
    GenerationMetrics,
    DomainInfo,
    DomainListResponse,
)


def test_seed_question_schema():
    seed = SeedQuestion(text="Calculate velocity", domain="Physics", difficulty=0.5)
    assert seed.text == "Calculate velocity"
    assert seed.domain == "Physics"
    assert seed.difficulty == 0.5
    assert isinstance(seed.metadata, dict)


def test_generation_request_schema():
    req = GenerationRequest(seed_question="Explain bubble sort", domain="Computer Science", count=60)
    assert req.seed_question == "Explain bubble sort"
    assert req.domain == "Computer Science"
    assert req.count == 60


def test_generation_request_validation():
    with pytest.raises(ValidationError):
        GenerationRequest(seed_question="Test", domain="Math", count=-5)


def test_generation_response_schema():
    var1 = VariationOutput(question="Var 1", answer_key="Ans 1", difficulty=0.4)
    var2 = VariationOutput(question="Var 2", answer_key="Ans 2", difficulty=0.6)
    res = GenerationResponse(variations=[var1, var2], duplicate_rate=0.05)
    
    data = res.model_dump()
    assert "variations" in data
    assert "duplicate_rate" in data
    assert len(data["variations"]) == 2
    assert data["variations"][0]["question"] == "Var 1"


def test_question_variation_schema():
    var = QuestionVariation(
        question="What is the speed?",
        answer_key="50 m/s",
        difficulty=0.5,
        domain="Physics"
    )
    assert var.id is not None
    assert var.question == "What is the speed?"
    assert var.answer_key == "50 m/s"
    assert var.confidence_score == 1.0


def test_difficulty_score_schema():
    ds = DifficultyScore(score=0.7, blooms_level="Apply", complexity_factors={"word_count": 25})
    assert ds.score == 0.7
    assert ds.blooms_level == "Apply"
    assert ds.complexity_factors["word_count"] == 25


def test_validation_result_schema():
    ds = DifficultyScore(score=0.5)
    vr = ValidationResult(
        is_valid=True,
        difficulty_score=ds,
        is_difficulty_equivalent=True,
        answer_valid=True
    )
    assert vr.is_valid is True
    assert vr.is_difficulty_equivalent is True
    assert len(vr.reasons) == 0


def test_duplicate_result_schema():
    dr = DuplicateResult(is_duplicate=True, similarity_score=0.92, matched_question_id="q123")
    assert dr.is_duplicate is True
    assert dr.similarity_score == 0.92
    assert dr.matched_question_id == "q123"


def test_answer_key_schema():
    ak = AnswerKey(
        question_text="Calculate force",
        answer_text="F = 100 N",
        explanation="F = m * a = 10 * 10 = 100",
        rubric_points=["Formula (50%)", "Calculation (50%)"]
    )
    assert ak.answer_text == "F = 100 N"
    assert len(ak.rubric_points) == 2


def test_review_item_schema():
    var = QuestionVariation(question="Sample Q", answer_key="Sample Ans", difficulty=0.6)
    ri = ReviewItem(variation=var, reason="Borderline difficulty", confidence_score=0.7)
    assert ri.id is not None
    assert ri.status == "pending"
    assert ri.confidence_score == 0.7


def test_generation_metrics_schema():
    gm = GenerationMetrics(
        requested_count=60,
        generated_count=65,
        accepted_count=60,
        rejected_count=5,
        duplicate_count=2,
        duplicate_rate=0.0308,
        low_confidence_count=1,
        generation_time_seconds=1.25
    )
    assert gm.requested_count == 60
    assert gm.accepted_count == 60
    assert gm.duplicate_count == 2
