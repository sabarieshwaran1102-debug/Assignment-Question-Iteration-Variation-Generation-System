"""
Unit tests for Generator module components.
"""

from packages.schemas.models import SeedQuestion
from packages.common.providers.mock_provider import MockLLMProvider
from apps.generator.parser import DefaultQuestionParser
from apps.generator.objective import DefaultLearningObjectiveAnalyzer
from apps.generator.answer import DefaultAnswerKeyGenerator
from apps.generator.variation import DefaultVariationGenerator


def test_question_parser():
    llm = MockLLMProvider()
    parser = DefaultQuestionParser(llm_provider=llm)
    seed = SeedQuestion(text="Calculate the acceleration of a 10kg mass when a force of 50N is applied.", domain="Physics")
    
    parsed = parser.parse(seed)
    assert parsed.raw_text == seed.text
    assert parsed.domain == "Physics"
    assert "v_1" in parsed.variables
    assert parsed.variables["v_1"] == 10.0
    assert parsed.variables["v_2"] == 50.0
    assert parsed.blooms_level in ["Apply", "Analyze", "Evaluate", "Understand"]


def test_learning_objective_analyzer():
    llm = MockLLMProvider()
    parser = DefaultQuestionParser(llm_provider=llm)
    analyzer = DefaultLearningObjectiveAnalyzer(llm_provider=llm)
    
    seed = SeedQuestion(text="Explain binary search time complexity.", domain="Computer Science")
    parsed = parser.parse(seed)
    objective = analyzer.extract_objective(parsed)
    
    assert objective.objective != ""
    assert objective.target_skill != ""
    assert "Computer Science" in objective.core_concepts


def test_answer_key_generator():
    llm = MockLLMProvider()
    answer_gen = DefaultAnswerKeyGenerator(llm_provider=llm)
    
    ak = answer_gen.generate_answer_key("Calculate force when m=5, a=10", domain="Physics", context={"index": 0})
    assert ak.answer_text != ""
    assert ak.explanation is not None


def test_variation_generator():
    llm = MockLLMProvider()
    answer_gen = DefaultAnswerKeyGenerator(llm_provider=llm)
    var_gen = DefaultVariationGenerator(llm_provider=llm, answer_generator=answer_gen)
    
    parser = DefaultQuestionParser(llm_provider=llm)
    analyzer = DefaultLearningObjectiveAnalyzer(llm_provider=llm)
    
    seed = SeedQuestion(text="Compute velocity of projectile 100m in 5s", domain="Physics")
    parsed = parser.parse(seed)
    objective = analyzer.extract_objective(parsed)
    
    variations = var_gen.generate_variations(seed=seed, parsed=parsed, objective=objective, count=10)
    assert len(variations) == 10
    
    # Verify variations are distinct and contain required fields
    questions_set = {v.question for v in variations}
    assert len(questions_set) == 10
    for v in variations:
        assert v.answer_key != ""
        assert 0.0 <= v.difficulty <= 1.0
