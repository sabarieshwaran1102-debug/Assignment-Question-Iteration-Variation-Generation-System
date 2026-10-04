"""
Unit tests for Orchestrator workflow pipeline.
"""

from packages.schemas.models import GenerationRequest
from packages.common.providers.mock_provider import MockLLMProvider, MockEmbeddingProvider, MockVectorStore
from apps.generator.parser import DefaultQuestionParser
from apps.generator.objective import DefaultLearningObjectiveAnalyzer
from apps.generator.answer import DefaultAnswerKeyGenerator
from apps.generator.variation import DefaultVariationGenerator
from apps.evaluator.difficulty import DefaultDifficultyAnalyzer, DefaultDifficultyEquivalenceValidator
from apps.evaluator.duplicate import DefaultDuplicateDetector
from apps.evaluator.answer_validator import DefaultAnswerValidator
from apps.evaluator.evaluator import DefaultVariationEvaluator
from apps.orchestrator.pipeline import GenerationOrchestrator


def test_orchestrator_workflow_60_variations():
    llm = MockLLMProvider()
    embedder = MockEmbeddingProvider(dimension=64)
    vector_store = MockVectorStore()

    parser = DefaultQuestionParser(llm_provider=llm)
    objective_analyzer = DefaultLearningObjectiveAnalyzer(llm_provider=llm)
    answer_gen = DefaultAnswerKeyGenerator(llm_provider=llm)
    generator = DefaultVariationGenerator(llm_provider=llm, answer_generator=answer_gen)

    diff_analyzer = DefaultDifficultyAnalyzer()
    equiv_validator = DefaultDifficultyEquivalenceValidator()
    dup_detector = DefaultDuplicateDetector(embedding_provider=embedder, vector_store=vector_store)
    ans_validator = DefaultAnswerValidator()

    evaluator = DefaultVariationEvaluator(
        difficulty_analyzer=diff_analyzer,
        equivalence_validator=equiv_validator,
        duplicate_detector=dup_detector,
        answer_validator=ans_validator
    )

    orchestrator = GenerationOrchestrator(
        parser=parser,
        objective_analyzer=objective_analyzer,
        generator=generator,
        evaluator=evaluator,
        duplicate_detector=dup_detector,
        embedding_provider=embedder,
        vector_store=vector_store
    )

    request = GenerationRequest(
        seed_question="Calculate the kinetic energy of a 2kg object moving at 10m/s.",
        domain="Physics",
        count=60
    )

    response, metrics, review_items = orchestrator.orchestrate_generation(request)

    # 1. Validate response matching PS8 contract
    assert len(response.variations) == 60
    assert isinstance(response.duplicate_rate, float)
    assert 0.0 <= response.duplicate_rate <= 1.0

    for item in response.variations:
        assert item.question != ""
        assert item.answer_key != ""
        assert 0.0 <= item.difficulty <= 1.0

    # 2. Validate metrics telemetry
    assert metrics.requested_count == 60
    assert metrics.accepted_count == 60
    assert metrics.generation_time_seconds > 0.0
