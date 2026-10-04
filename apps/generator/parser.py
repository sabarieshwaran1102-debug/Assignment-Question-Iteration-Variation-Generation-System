"""
Default Question Parser implementation.
Extracts mathematical parameters, problem structure, and concepts from seed questions.
"""

import re
from typing import Dict, Any, List
from packages.schemas.models import SeedQuestion
from packages.common.providers.interfaces import LLMProvider
from apps.generator.interfaces import QuestionParser, ParsedQuestion


class DefaultQuestionParser(QuestionParser):
    """Parses seed questions using LLMProvider or rule-based feature extraction."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def parse(self, seed: SeedQuestion) -> ParsedQuestion:
        # Rule-based fallback parameter extraction for deterministic speed
        extracted_nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", seed.text)]
        vars_dict: Dict[str, Any] = {f"v_{i+1}": val for i, val in enumerate(extracted_nums)}
        
        # Determine Bloom's taxonomy level heuristic
        text_lower = seed.text.lower()
        if any(w in text_lower for w in ["evaluate", "optimize", "compare", "critique"]):
            blooms = "Evaluate"
        elif any(w in text_lower for w in ["calculate", "compute", "solve", "find", "determine"]):
            blooms = "Apply"
        elif any(w in text_lower for w in ["explain", "describe", "summarize", "why"]):
            blooms = "Understand"
        else:
            blooms = "Analyze"

        # Determine problem type
        if vars_dict:
            prob_type = "quantitative analysis"
        elif any(w in text_lower for w in ["code", "algorithm", "function", "array"]):
            prob_type = "algorithmic reasoning"
        else:
            prob_type = "conceptual analysis"

        parsed = ParsedQuestion(
            raw_text=seed.text,
            domain=seed.domain,
            concept=f"Core concept in {seed.domain}",
            variables=vars_dict,
            problem_type=prob_type,
            constraints=["standard assumptions", "positive values"],
            blooms_level=blooms
        )
        return parsed
