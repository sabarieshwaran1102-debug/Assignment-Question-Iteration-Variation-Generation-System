"""
Interfaces for the Question Generator module.
Defines contracts for question parsing, objective extraction, variation generation, and answer key creation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from packages.schemas.models import SeedQuestion, QuestionVariation, AnswerKey


class ParsedQuestion(BaseModel):
    """Structured output from parsing a seed question."""
    raw_text: str
    domain: str
    concept: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    problem_type: str = Field(default="analytical")
    constraints: List[str] = Field(default_factory=list)
    blooms_level: str = Field(default="Apply")


class LearningObjective(BaseModel):
    """Extracted pedagogical learning objective."""
    objective: str
    target_skill: str
    blooms_level: str
    core_concepts: List[str] = Field(default_factory=list)


class QuestionParser(ABC):
    """Abstract interface for parsing seed questions into structured components."""

    @abstractmethod
    def parse(self, seed: SeedQuestion) -> ParsedQuestion:
        """Parse raw seed question text into structured components."""
        pass


class LearningObjectiveAnalyzer(ABC):
    """Abstract interface for extracting learning objectives from parsed questions."""

    @abstractmethod
    def extract_objective(self, parsed: ParsedQuestion) -> LearningObjective:
        """Extract core learning objective from parsed question."""
        pass


class VariationGenerator(ABC):
    """Abstract interface for generating question variations."""

    @abstractmethod
    def generate_variations(
        self,
        seed: SeedQuestion,
        parsed: ParsedQuestion,
        objective: LearningObjective,
        count: int,
        target_domain: Optional[str] = None,
        start_index: int = 0,
        blueprints: Optional[List[Any]] = None
    ) -> List[QuestionVariation]:
        """Generate N question variations preserving learning objective."""
        pass



class AnswerKeyGenerator(ABC):
    """Abstract interface for generating structured answer keys for variations."""

    @abstractmethod
    def generate_answer_key(self, variation_text: str, domain: str, context: Optional[Dict[str, Any]] = None) -> AnswerKey:
        """Generate detailed answer key and step-by-step solution for a variation."""
        pass
