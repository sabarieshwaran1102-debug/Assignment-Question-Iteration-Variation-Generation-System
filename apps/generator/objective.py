"""
Default Learning Objective Analyzer implementation.
Identifies pedagogical objectives and core skills required to solve the question.
"""

from packages.common.providers.interfaces import LLMProvider
from apps.generator.interfaces import LearningObjectiveAnalyzer, ParsedQuestion, LearningObjective


class DefaultLearningObjectiveAnalyzer(LearningObjectiveAnalyzer):
    """Analyzes parsed questions to extract learning objectives."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def extract_objective(self, parsed: ParsedQuestion) -> LearningObjective:
        objective_text = f"Demonstrate mastery of {parsed.problem_type} in {parsed.domain}."
        skill_text = f"{parsed.blooms_level} domain principles of {parsed.domain} to solve complex scenarios."
        
        return LearningObjective(
            objective=objective_text,
            target_skill=skill_text,
            blooms_level=parsed.blooms_level,
            core_concepts=[parsed.concept, parsed.domain]
        )
