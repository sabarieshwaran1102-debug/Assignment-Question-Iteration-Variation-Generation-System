"""
Default Answer Key Generator implementation.
Generates structured answers, explanations, and grading rubrics for question variations.
"""

from typing import Dict, Optional, Any
from packages.schemas.models import AnswerKey, VariationBlueprint
from packages.common.providers.interfaces import LLMProvider
from apps.generator.interfaces import AnswerKeyGenerator
from apps.generator.solver import DefaultAnswerSolver


class DefaultAnswerKeyGenerator(AnswerKeyGenerator):
    """Generates structured answer keys for generated question variations using deterministic solver."""

    def __init__(self, llm_provider: LLMProvider, solver: Optional[Any] = None):
        self.llm_provider = llm_provider
        self.solver = solver or DefaultAnswerSolver()

    def generate_answer_key(self, variation_text: str, domain: str, context: Optional[Dict[str, Any]] = None) -> AnswerKey:
        ctx = context or {}
        blueprint = ctx.get("blueprint")

        # If blueprint is provided, compute answer deterministically via AnswerSolver
        if isinstance(blueprint, VariationBlueprint):
            ans_key = self.solver.solve(blueprint)
            ans_key.question_text = variation_text
            return ans_key

        # If blueprint variables are passed in context
        if "distance" in ctx or "variables" in ctx:
            vars_dict = ctx.get("variables", {})
            d_val = ctx.get("distance", vars_dict.get("distance", 100.0))
            t_val = ctx.get("time", vars_dict.get("time", 5.0))
            bp = VariationBlueprint(
                domain=domain,
                topic="Kinematics",
                learning_objective="Calculate velocity using distance and time",
                formula="v = d / t",
                scenario=ctx.get("scenario", "vehicle"),
                known_variables={"distance": float(d_val), "time": float(t_val)},
                units={"distance": "m", "time": "s", "velocity": "m/s"}
            )
            ans_key = self.solver.solve(bp)
            ans_key.question_text = variation_text
            return ans_key

        # Fallback to LLM provider
        index = ctx.get("index", 0)
        return self.llm_provider.generate_structured(
            prompt=variation_text,
            schema=AnswerKey,
            domain=domain,
            index=index
        )

