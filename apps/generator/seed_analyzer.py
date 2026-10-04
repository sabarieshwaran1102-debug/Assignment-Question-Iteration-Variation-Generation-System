"""
Seed Analyzer module.
Performs deep structural and cognitive analysis on seed questions, producing structured SeedQuestionAnalysis instances.
"""

from abc import ABC, abstractmethod
from typing import Optional
import re

from packages.schemas.models import SeedQuestion, SeedQuestionAnalysis
from packages.common.providers.interfaces import LLMProvider
from apps.generator.interfaces import QuestionParser, LearningObjectiveAnalyzer


class SeedAnalyzer(ABC):
    """Abstract interface for seed question analysis."""

    @abstractmethod
    def analyze_seed(self, seed: SeedQuestion) -> SeedQuestionAnalysis:
        """Extract structured cognitive and structural analysis from seed question."""
        pass


class DefaultSeedAnalyzer(SeedAnalyzer):
    """Default Seed Analyzer utilizing QuestionParser, LearningObjectiveAnalyzer, and Bloom heuristics."""

    def __init__(
        self,
        parser: QuestionParser,
        objective_analyzer: LearningObjectiveAnalyzer,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.parser = parser
        self.objective_analyzer = objective_analyzer
        self.llm_provider = llm_provider

    def analyze_seed(self, seed: SeedQuestion) -> SeedQuestionAnalysis:
        parsed = self.parser.parse(seed)
        obj = self.objective_analyzer.extract_objective(parsed)

        # Detect subtopic heuristics
        text_lower = seed.text.lower()
        if any(w in text_lower for w in ["velocity", "speed", "distance", "time", "moving"]):
            subtopic = "Kinematics & Motion"
            formula = "v = d / t"
        elif any(w in text_lower for w in ["sort", "array", "binary tree", "algorithm"]):
            subtopic = "Algorithmic Analysis"
            formula = "Time Complexity Analysis"
        elif any(w in text_lower for w in ["voltage", "current", "resistor"]):
            subtopic = "DC Circuits"
            formula = "V = I * R"
        else:
            subtopic = f"Core Subtopic in {seed.domain}"
            formula = "Standard Domain Rule"

        # Determine Bloom's Taxonomy classification
        blooms = obj.blooms_level or parsed.blooms_level or "Apply"

        baseline_diff = seed.difficulty if seed.difficulty is not None else 0.50

        analysis = SeedQuestionAnalysis(
            domain=seed.domain,
            topic="Kinematics" if "velocity" in text_lower else parsed.concept,
            subtopic=subtopic,
            learning_objective=obj.objective,
            bloom_level=blooms,
            difficulty=baseline_diff,
            question_type=parsed.problem_type or "quantitative calculation",
            concepts=[parsed.concept, seed.domain, subtopic],
            variables=parsed.variables,
            constraints=parsed.constraints or ["positive numerical values"],
            solution_method=f"Direct formula application ({formula})",
            formulas=[formula]
        )
        return analysis
