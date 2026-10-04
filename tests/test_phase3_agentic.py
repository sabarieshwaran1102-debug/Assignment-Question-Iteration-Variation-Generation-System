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


def test_velocity_seed_regression_authoritative_blueprint_and_answer_key_matching(phase3_master_agent):
    """
    Regression test for Phase 3.1:
    Verifies that a velocity seed question ("Calculate velocity when distance is 100m and time is 5 seconds"):
    1. Generates velocity calculation questions (v = d / t) without introducing unrelated physics topics.
    2. Has exact numerical alignment between the distance and time in the question text and the calculated answer key.
    3. Maintains authoritative blueprint ownership from VariationPlanner.
    """
    import re

    req = GenerationRequest(
        seed_question="Calculate the velocity of a vehicle moving 100m in 5 seconds.",
        domain="Physics",
        count=10
    )

    resp, metrics, reviews = phase3_master_agent.orchestrate_generation(req)

    assert len(resp.variations) == 10

    unrelated_keywords = ["inclined plane", "orbit", "cyclotron", "mass sliding", "friction coefficient"]

    for var in resp.variations:
        q_text = var.question
        ans_key = var.answer_key

        # 1. Verify no unrelated physics domain contamination
        for kw in unrelated_keywords:
            assert kw not in q_text.lower(), f"Unrelated physics topic '{kw}' found in generated question: '{q_text}'"

        # 2. Extract distance and time numbers from question text
        numbers = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", q_text)]
        assert len(numbers) >= 2, f"Question text does not contain distance and time parameters: '{q_text}'"

        dist_val = numbers[0]
        time_val = numbers[1]
        expected_v = dist_val / time_val

        # 3. Verify answer key matches expected velocity
        if expected_v.is_integer():
            expected_str = f"{int(expected_v)} m/s"
        else:
            expected_str = f"{expected_v:.2f} m/s"

        assert expected_str in ans_key or f"{expected_v:.1f} m/s" in ans_key, (
            f"Answer key '{ans_key}' does not match expected calculated velocity '{expected_str}' "
            f"for question parameters distance={dist_val}, time={time_val}"
        )


def test_sixty_variations_diversity_difficulty_and_telemetry(phase3_master_agent):
    """
    Phase 3.2 Test:
    Verifies that generating 60 variations:
    1. Produces 60 distinct variations across multiple variation strategies (not 5 repeating templates).
    2. Uses real difficulty analysis rather than an artificial cyclic difficulty sequence (0.4, 0.45, 0.5, 0.55, 0.6).
    3. Populates all Phase 3.2 benchmark instrumentation telemetry in GenerationMetrics.
    """
    req = GenerationRequest(
        seed_question="Calculate the velocity of a vehicle moving 100m in 5 seconds.",
        domain="Physics",
        count=60
    )

    resp, metrics, reviews = phase3_master_agent.orchestrate_generation(req)

    # 1. Verify 60 accepted variations
    assert len(resp.variations) == 60
    assert metrics.requested_count == 60
    assert metrics.accepted_count == 60

    # 2. Verify Telemetry Instrumentation
    assert metrics.total_wall_clock_time >= 0.0
    assert metrics.objective_preservation_rate >= 0.90
    assert metrics.answer_key_correctness_rate >= 0.90
    assert metrics.difficulty_equivalence_rate >= 0.80
    assert isinstance(metrics.variation_strategy_distribution, dict)
    assert len(metrics.variation_strategy_distribution) >= 3

    # 3. Verify high structural diversity (not 5 repeated templates)
    unique_questions = set(v.question for v in resp.variations)
    assert len(unique_questions) >= 55, f"Expected at least 55 unique questions out of 60, got {len(unique_questions)}"

    # 4. Verify difficulty values come from analyzer and do not repeat in a 5-item artificial cycle (0.4, 0.45, 0.5, 0.55, 0.6)
    diffs = [round(v.difficulty, 3) for v in resp.variations]
    cyclic_pattern = [0.4, 0.45, 0.5, 0.55, 0.6]
    first_five = diffs[:5]
    next_five = diffs[5:10]
    assert not (first_five == cyclic_pattern and next_five == cyclic_pattern), (
        "Difficulty values still match the artificial repeating cycle (0.4, 0.45, 0.5, 0.55, 0.6)"
    )


def test_variation_41_42_regression_rejection_of_hallucinated_concepts():
    """
    Regression Test for Phase 3.2.1:
    Verifies that hallucinated or inconsistent physics concepts (e.g. Mach number, shockwave angle,
    magnetic drive force, ballistic pendulum) generated for a v = d / t velocity blueprint are strictly
    rejected by DefaultBlueprintQuestionConsistencyValidator and SpecializedQualityGate.
    """
    from apps.evaluator.blueprint_validator import DefaultBlueprintQuestionConsistencyValidator
    from packages.schemas.models import QuestionVariation, VariationBlueprint

    validator = DefaultBlueprintQuestionConsistencyValidator()

    # Blueprint for simple v = d / t kinematics
    bp = VariationBlueprint(
        domain="Physics",
        topic="Kinematics",
        target_variable="velocity",
        formula="v = d / t",
        known_variables={"distance": 1600.0, "time": 40.0},
        scenario="supersonic jet flying a test route",
        strategy_name="direct_calculation",
        learning_objective="Calculate velocity using v = d / t"
    )

    # Candidate 41 representation with hallucinated Mach number / shockwave angle
    bad_cand_41 = QuestionVariation(
        id="var_41",
        seed_question_id="seed_1",
        question="For supersonic jet breaking the sound barrier where mach number = 1600.00 and secondary parameter = 40.00, calculate the flight speed and shockwave angle.",
        answer_key="40.00 m/s\nExplanation: Step 1: Given distance = 1600.0 m, time = 40.0 s. Step 2: v = d / t = 40.00 m/s.",
        difficulty=0.5,
        domain="Physics",
        learning_objective="Calculate velocity using v = d / t",
        context_changes=bp.model_dump_json(),
        metadata={"blueprint": bp}
    )

    is_valid_41, reasons_41 = validator.validate_blueprint_consistency(bad_cand_41, blueprint=bp)
    assert is_valid_41 is False
    assert any("mach number" in r.lower() or "shockwave angle" in r.lower() or "inconsistent" in r.lower() for r in reasons_41)

    # Candidate 42 representation with hallucinated magnetic drive force / acceleration
    bad_cand_42 = QuestionVariation(
        id="var_42",
        seed_question_id="seed_1",
        question="For magnetic levitation train accelerating smoothly where magnetic drive force = 1050.00 and time = 21.00, calculate acceleration and peak velocity.",
        answer_key="50.00 m/s\nExplanation: Step 1: Given distance = 1050.0 m, time = 21.0 s. Step 2: v = d / t = 50.00 m/s.",
        difficulty=0.5,
        domain="Physics",
        learning_objective="Calculate velocity using v = d / t",
        context_changes=bp.model_dump_json(),
        metadata={"blueprint": bp}
    )

    is_valid_42, reasons_42 = validator.validate_blueprint_consistency(bad_cand_42, blueprint=bp)
    assert is_valid_42 is False
    assert any("magnetic drive" in r.lower() or "acceleration" in r.lower() or "inconsistent" in r.lower() for r in reasons_42)



