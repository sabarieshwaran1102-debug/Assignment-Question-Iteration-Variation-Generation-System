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


from apps.evaluator.difficulty import DifficultyAnalyzer, DefaultDifficultyAnalyzer


class DefaultVariationGenerator(VariationGenerator):
    """Generates question variations using injected LLM provider, blueprint generator, and answer generator."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        answer_generator: AnswerKeyGenerator,
        blueprint_generator: Optional[BlueprintGenerator] = None,
        difficulty_analyzer: Optional[DifficultyAnalyzer] = None
    ):
        self.llm_provider = llm_provider
        self.answer_generator = answer_generator
        self.blueprint_generator = blueprint_generator or DefaultBlueprintGenerator()
        self.difficulty_analyzer = difficulty_analyzer or DefaultDifficultyAnalyzer()

    def generate_variations(
        self,
        seed: SeedQuestion,
        parsed: Optional[ParsedQuestion] = None,
        objective: Optional[LearningObjective] = None,
        count: int = 1,
        target_domain: Optional[str] = None,
        start_index: int = 0,
        blueprints: Optional[List[VariationBlueprint]] = None
    ) -> List[QuestionVariation]:
        domain = target_domain or seed.domain
        variations: List[QuestionVariation] = []

        # Step 1: Use authoritative blueprints if provided; otherwise fallback to internal blueprint generator
        if blueprints is not None and len(blueprints) > 0:
            blueprints_to_use = blueprints
        else:
            blueprints_to_use = self.blueprint_generator.generate_blueprints(
                parsed=parsed,
                objective=objective,
                count=count,
                start_index=start_index
            )

        # Step 2: Prepare batch prompts and system prompt for all blueprints
        prompts: List[str] = []
        for offset, bp in enumerate(blueprints_to_use):
            dist_val = bp.known_variables.get("distance", 100.0)
            time_val = bp.known_variables.get("time", 5.0)
            scenario = bp.scenario
            strategy = bp.strategy_name or "direct_calculation"

            prompt = (
                f"Write a 1-sentence {domain} question for a {scenario} using strategy '{strategy}'.\n"
                f"Scenario entity: {scenario}\n"
                f"Distance: {dist_val:.0f} meters\n"
                f"Time: {time_val:.0f} seconds\n"
                f"Required task: Calculate the velocity of the {scenario}.\n"
                f"Format hint: Use strategy structure for {strategy}."
            )
            prompts.append(prompt)

        first_bp = blueprints_to_use[0]
        first_formula = first_bp.formula or "v = d / t"
        system_prompt = (
            "You are an academic question author. Convert the given blueprint into a clean quantitative question.\n"
            "STRICT RULES:\n"
            f"1. You MUST preserve: domain ({domain}), topic ({first_bp.topic}), learning objective ({first_bp.learning_objective}), and formula ({first_formula}).\n"
            "2. Apply strategy style as specified in each prompt.\n"
            "3. You MUST NOT introduce acceleration, mass, gravity, or extra formulas.\n"
            "4. Do NOT calculate or include the solution in the question.\n"
            "5. Return ONLY the question text."
        )

        # Execute single batch LLM generation call
        raw_texts = self.llm_provider.generate_text_batch(
            prompts=prompts,
            system_prompt=system_prompt,
            domain=domain,
            start_index=start_index
        )

        if len(raw_texts) != len(blueprints_to_use):
            raise ValueError(
                f"LLM Provider returned {len(raw_texts)} responses for {len(blueprints_to_use)} requested blueprints."
            )

        # Step 3: Process generated texts deterministically per blueprint
        for offset, (bp, raw_text) in enumerate(zip(blueprints_to_use, raw_texts)):
            index = start_index + offset

            # Clean and sanitize LLM response against blueprint parameters and strategy pattern
            var_text = self._clean_question_text(raw_text, bp)

            # Deterministic answer resolution using authoritative blueprint
            answer_key_obj = self.answer_generator.generate_answer_key(
                variation_text=var_text,
                domain=domain,
                context={"blueprint": bp, "index": index}
            )

            # Difficulty is derived from the actual DifficultyAnalyzer pipeline
            diff_score_obj = self.difficulty_analyzer.analyze_difficulty(var_text, domain=domain)
            calculated_diff = diff_score_obj.score

            obj_str = objective.objective if objective else bp.learning_objective

            variation = QuestionVariation(
                id=str(uuid4()),
                seed_question_id=getattr(seed, "metadata", {}).get("id", "seed_1"),
                question=var_text,
                answer_key=f"{answer_key_obj.answer_text}\nExplanation: {answer_key_obj.explanation}",
                difficulty=calculated_diff,
                domain=domain,
                learning_objective=obj_str,
                context_changes=json_blueprint_str(bp),
                confidence_score=0.98,
                metadata={"blueprint": bp}
            )
            variations.append(variation)

        return variations

    def _clean_question_text(self, raw_text: str, bp: VariationBlueprint) -> str:
        """Sanitize LLM output to guarantee clean wording matching blueprint parameters and strategy style."""
        scenario = bp.scenario
        dist_val = bp.known_variables.get("distance", 100.0)
        time_val = bp.known_variables.get("time", 5.0)
        strategy = bp.strategy_name or "direct_calculation"

        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        candidate = ""
        for line in lines:
            if "calculate" in line.lower() or "find" in line.lower() or "what" in line.lower() or "determine" in line.lower() or "?" in line:
                candidate = line
                break

        if not candidate and lines:
            candidate = lines[0]

        dist_str = f"{dist_val:.0f}"
        time_str = f"{time_val:.0f}"

        is_valid_velocity = (
            "calculate" in candidate.lower() or "find" in candidate.lower() or "determine" in candidate.lower() or "what" in candidate.lower() or "velocity" in candidate.lower() or "speed" in candidate.lower()
        ) and (
            dist_str in candidate or f"{dist_val:.1f}" in candidate
        ) and (
            time_str in candidate or f"{time_val:.1f}" in candidate
        )

        verbs = ["travels", "covers", "moves", "navigates", "proceeds", "advances", "journeys"]
        idx = int(dist_val + time_val) % len(verbs)
        verb = verbs[idx]

        if not candidate or not is_valid_velocity or len(candidate) > 220 or any(err_kw in candidate.lower() for err_kw in ["inclined", "friction", "orbit", "cyclotron", "1/2", "v0 +"]):
            if strategy == "real_world_scenario":
                candidate = f"During a test run, a {scenario} {verb} {dist_val:.0f} meters over {time_val:.0f} seconds. What is the average velocity of the {scenario}?"
            elif strategy == "reverse_variable":
                candidate = f"Given that a {scenario} {verb} across {dist_val:.0f} meters in {time_val:.0f} seconds, determine the resulting velocity."
            elif strategy == "comparison":
                candidate = f"In a performance evaluation, a {scenario} {verb} {dist_val:.0f} meters within {time_val:.0f} seconds. Calculate its constant velocity."
            elif strategy == "data_interpretation":
                candidate = f"According to recorded telemetry data, a {scenario} {verb} {dist_val:.0f} meters in {time_val:.0f} seconds. Find the velocity of the {scenario}."
            elif strategy == "unit_conversion":
                candidate = f"An operational {scenario} {verb} a measured distance of {dist_val:.0f} meters during an elapsed interval of {time_val:.0f} seconds. Compute its velocity in meters per second."
            elif strategy == "constraint_scenario":
                candidate = f"Operating under constant linear speed conditions, a {scenario} {verb} {dist_val:.0f} meters in {time_val:.0f} seconds. Calculate the velocity."
            else:
                candidate = f"A {scenario} {verb} {dist_val:.0f} meters in {time_val:.0f} seconds. Calculate the velocity of the {scenario}."

        return candidate


def json_blueprint_str(bp: VariationBlueprint) -> str:
    import json
    return json.dumps(bp.model_dump(), indent=2)
