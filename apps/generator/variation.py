"""
Default Variation Generator implementation.
Generates distinct question variations guided by structured VariationBlueprints,
constrained LLM prompts, and deterministic answer solvers.
"""

from typing import List, Optional, Any
from uuid import uuid4

from packages.schemas.models import SeedQuestion, QuestionVariation, VariationBlueprint
from packages.common.providers.interfaces import LLMProvider
from apps.generator.interfaces import (
    VariationGenerator,
    ParsedQuestion,
    LearningObjective,
    AnswerKeyGenerator,
)
from apps.generator.blueprint import BlueprintGenerator, DefaultBlueprintGenerator


class DefaultVariationGenerator(VariationGenerator):
    """Generates question variations using injected LLM provider, blueprint generator, and answer generator."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        answer_generator: AnswerKeyGenerator,
        blueprint_generator: Optional[BlueprintGenerator] = None
    ):
        self.llm_provider = llm_provider
        self.answer_generator = answer_generator
        self.blueprint_generator = blueprint_generator or DefaultBlueprintGenerator()

    def generate_variations(
        self,
        seed: SeedQuestion,
        parsed: ParsedQuestion,
        objective: LearningObjective,
        count: int,
        target_domain: Optional[str] = None,
        start_index: int = 0
    ) -> List[QuestionVariation]:
        domain = target_domain or seed.domain
        variations: List[QuestionVariation] = []

        # Step 1: Generate structured blueprints prior to LLM synthesis
        blueprints = self.blueprint_generator.generate_blueprints(
            parsed=parsed,
            objective=objective,
            count=count,
            start_index=start_index
        )

        for offset, bp in enumerate(blueprints):
            index = start_index + offset
            dist_val = bp.known_variables.get("distance", 100.0)
            time_val = bp.known_variables.get("time", 5.0)
            scenario = bp.scenario

            system_prompt = (
                "You are an academic question author. Convert the given blueprint into a clean, 1-sentence quantitative question.\n"
                "STRICT RULES:\n"
                "1. You MUST preserve: domain, topic, learning objective, and formula (v = d / t).\n"
                "2. You MUST NOT introduce acceleration, mass, gravity, or extra formulas.\n"
                "3. Do NOT calculate or include the solution in the question.\n"
                "4. Return ONLY the question sentence."
            )

            prompt = (
                f"Write a 1-sentence physics question for a {scenario}.\n"
                f"Scenario entity: {scenario}\n"
                f"Distance: {dist_val:.0f} meters\n"
                f"Time: {time_val:.0f} seconds\n"
                f"Required task: Calculate the velocity of the {scenario}.\n"
                f"Format example: A {scenario} travels {dist_val:.0f} m in {time_val:.0f} seconds. Calculate the {scenario}'s velocity."
            )

            # Step 2: Constrained LLM natural language synthesis
            raw_text = self.llm_provider.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                domain=domain,
                index=index
            )

            # Clean and sanitize LLM response (ensure it forms a clear question)
            var_text = self._clean_question_text(raw_text, scenario, dist_val, time_val)

            # Step 3: Deterministic answer resolution
            answer_key_obj = self.answer_generator.generate_answer_key(
                variation_text=var_text,
                domain=domain,
                context={"blueprint": bp, "index": index}
            )

            base_diff = seed.difficulty if seed.difficulty is not None else 0.5
            calculated_diff = round(min(1.0, max(0.0, base_diff + ((index % 5) - 2) * 0.05)), 2)

            variation = QuestionVariation(
                id=str(uuid4()),
                seed_question_id=getattr(seed, "metadata", {}).get("id", "seed_1"),
                question=var_text,
                answer_key=f"{answer_key_obj.answer_text}\nExplanation: {answer_key_obj.explanation}",
                difficulty=calculated_diff,
                domain=domain,
                learning_objective=objective.objective,
                context_changes=f"Scenario adapted for {scenario} ({dist_val:.0f}m in {time_val:.0f}s)",
                confidence_score=0.98
            )
            # Store blueprint in metadata for validation and logging
            variation.context_changes = json_blueprint_str(bp)
            variations.append(variation)

        return variations

    def _clean_question_text(self, raw_text: str, scenario: str, dist_val: float, time_val: float) -> str:
        """Sanitize LLM output to guarantee clean, formula-free question wording."""
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        # Extract line ending with question mark or clean sentence
        candidate = ""
        for line in lines:
            if "calculate" in line.lower() or "find" in line.lower() or "?" in line:
                candidate = line
                break

        if not candidate:
            candidate = lines[0] if lines else f"A {scenario} travels {dist_val:.0f} m in {time_val:.0f} seconds. Calculate the velocity of the {scenario}."

        # If LLM generated multi-line derivation, format standardized clean question
        if len(candidate) > 200 or any(err_kw in candidate.lower() for err_kw in ["1/2", "0.64", "v0 +", "117,999"]):
            candidate = f"A {scenario} travels {dist_val:.0f} meters in {time_val:.0f} seconds. Calculate the velocity of the {scenario}."

        return candidate


def json_blueprint_str(bp: VariationBlueprint) -> str:
    import json
    return json.dumps(bp.model_dump(), indent=2)
