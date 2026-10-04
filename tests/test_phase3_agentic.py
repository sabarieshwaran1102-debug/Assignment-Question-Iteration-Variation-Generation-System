"""
Phase 3 Unit and Integration Test Suite.
Tests SeedAnalyzer, BloomClassifier/Evaluator, VariationPlanner, RAG Layer, AnswerSolver,
MasterAgent, Low-Confidence Queue, CSV/JSON export, and Fine-Tuning Dataset generation.
"""

import os
import json
import pytest
from packages.schemas.models import SeedQuestion, GenerationRequest
from packages.common.providers.mock_provider import MockLLMProvider, MockEmbeddingProvider, MockVectorStore
from packages.common.rag import DefaultKnowledgeContextProvider
from apps.generator.parser import DefaultQuestionParser
from apps.generator.objective import DefaultLearningObjectiveAnalyzer
from apps.generator.seed_analyzer import DefaultSeedAnalyzer
from apps.generator.planner import DefaultVariationPlanner
from apps.generator.blueprint import DefaultBlueprintGenerator
from apps.generator.solver import DefaultAnswerSolver
from apps.generator.answer import DefaultAnswerKeyGenerator
from apps.generator.variation import DefaultVariationGenerator

from apps.evaluator.bloom import DefaultBloomClassifier, DefaultBloomEvaluator
from apps.evaluator.difficulty import DefaultDifficultyAnalyzer, DefaultDifficultyEquivalenceValidator
from apps.evaluator.duplicate import DefaultDuplicateDetector
from apps.evaluator.answer_validator import DefaultAnswerValidator
from apps.evaluator.objective_validator import DefaultLearningObjectiveConsistencyValidator
from apps.evaluator.specialized import SpecializedQualityGate

from apps.orchestrator.agent import MasterAgent


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def phase3_master_agent(mock_llm):
    parser = DefaultQuestionParser(mock_llm)
    obj_analyzer = DefaultLearningObjectiveAnalyzer(mock_llm)
    seed_analyzer = DefaultSeedAnalyzer(parser, obj_analyzer, mock_llm)

    knowledge_provider = DefaultKnowledgeContextProvider()
    planner = DefaultVariationPlanner()
    blueprint_gen = DefaultBlueprintGenerator()
    solver = DefaultAnswerSolver()

    answer_gen = DefaultAnswerKeyGenerator(mock_llm, solver)
    generator = DefaultVariationGenerator(mock_llm, answer_gen, blueprint_gen)

    diff_an = DefaultDifficultyAnalyzer()
    equiv_val = DefaultDifficultyEquivalenceValidator()
    embedder = MockEmbeddingProvider()
    vstore = MockVectorStore()
    dup_det = DefaultDuplicateDetector(embedder, vstore)
    ans_val = DefaultAnswerValidator()
    obj_val = DefaultLearningObjectiveConsistencyValidator()
    bloom_eval = DefaultBloomEvaluator(classifier=DefaultBloomClassifier())

    quality_gate = SpecializedQualityGate(
        objective_validator=obj_val,
        answer_validator=ans_val,
        difficulty_analyzer=diff_an,
        equivalence_validator=equiv_val,
        duplicate_detector=dup_det,
        bloom_evaluator=bloom_eval
    )

    agent = MasterAgent(
        seed_analyzer=seed_analyzer,
        knowledge_provider=knowledge_provider,
        planner=planner,
        generator=generator,
        answer_solver=solver,
        answer_generator=answer_gen,
        quality_gate=quality_gate,
        duplicate_detector=dup_det,
        embedding_provider=embedder,
        vector_store=vstore,
        dataset_path="scratch/test_finetuning.jsonl"
    )
    return agent


def test_seed_analyzer(mock_llm):
    parser = DefaultQuestionParser(mock_llm)
    obj_analyzer = DefaultLearningObjectiveAnalyzer(mock_llm)
    analyzer = DefaultSeedAnalyzer(parser, obj_analyzer, mock_llm)

    seed = SeedQuestion(text="Calculate the velocity of a vehicle moving 100m in 5 seconds.", domain="Physics")
    analysis = analyzer.analyze_seed(seed)

    assert analysis.domain == "Physics"
    assert analysis.topic == "Kinematics"
    assert "v = d / t" in analysis.formulas
    assert analysis.bloom_level == "Apply"


def test_bloom_classifier_and_evaluator():
    classifier = DefaultBloomClassifier()
    evaluator = DefaultBloomEvaluator(classifier=classifier)

    text_apply = "Calculate the speed of a cyclist traveling 240 m in 12 seconds."
    text_remember = "Define the term velocity in classical mechanics."

    assert classifier.classify_bloom(text_apply) == "Apply"
    assert classifier.classify_bloom(text_remember) == "Remember"

    res = evaluator.evaluate_bloom_equivalence(seed_level="Apply", candidate_text=text_apply)
    assert res.passed is True

    res_distant = evaluator.evaluate_bloom_equivalence(seed_level="Create", candidate_text=text_remember)
    assert res_distant.passed is False


def test_variation_planner(mock_llm):
    parser = DefaultQuestionParser(mock_llm)
    obj_analyzer = DefaultLearningObjectiveAnalyzer(mock_llm)
    analyzer = DefaultSeedAnalyzer(parser, obj_analyzer, mock_llm)
    planner = DefaultVariationPlanner()

    seed = SeedQuestion(text="Calculate the velocity of a vehicle moving 100m in 5 seconds.", domain="Physics")
    analysis = analyzer.analyze_seed(seed)
    blueprints = planner.plan_variations(analysis, count=5)

    assert len(blueprints) == 5
    assert blueprints[0].formula == "v = d / t"
    assert blueprints[0].known_variables["distance"] > 0


def test_rag_knowledge_context():
    provider = DefaultKnowledgeContextProvider()
    ctx = provider.get_context(domain="Physics", topic="Kinematics")

    assert ctx.taxonomy["domain"] == "Physics"
    assert len(ctx.benchmark_examples) > 0
    assert "Apply" in ctx.bloom_definitions


def test_deterministic_answer_solver():
    solver = DefaultAnswerSolver()
    parser = DefaultQuestionParser(MockLLMProvider())
    obj_analyzer = DefaultLearningObjectiveAnalyzer(MockLLMProvider())
    analyzer = DefaultSeedAnalyzer(parser, obj_analyzer)
    planner = DefaultVariationPlanner()

    seed = SeedQuestion(text="Calculate the velocity of a vehicle moving 100m in 5 seconds.", domain="Physics")
    blueprints = planner.plan_variations(analyzer.analyze_seed(seed), count=1)

    key = solver.solve(blueprints[0])
    assert "20 m/s" in key.answer_text
    assert "v = d / t" in key.explanation


def test_master_agent_execution(phase3_master_agent):
    req = GenerationRequest(
        seed_question="Calculate the velocity of a vehicle moving 100m in 5 seconds.",
        domain="Physics",
        count=10
    )

    resp, metrics, reviews = phase3_master_agent.orchestrate_generation(req)

    assert len(resp.variations) == 10
    assert metrics.accepted_count == 10
    assert metrics.duplicate_rate < 0.10
    assert metrics.generation_time_seconds >= 0.0


def test_bulk_export_csv_and_json(phase3_master_agent, tmp_path):
    req = GenerationRequest(
        seed_question="Calculate the velocity of a vehicle moving 100m in 5 seconds.",
        domain="Physics",
        count=3
    )
    resp, metrics, reviews = phase3_master_agent.orchestrate_generation(req)

    csv_file = str(tmp_path / "export.csv")
    json_file = str(tmp_path / "export.json")

    phase3_master_agent.export_csv(resp.variations, csv_file)
    phase3_master_agent.export_json(resp.variations, json_file)

    assert os.path.exists(csv_file)
    assert os.path.exists(json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 3
