"""
Variation Planner module.
Creates diverse, controlled VariationBlueprints applying strategy patterns
(direct calculation, real-world scenario, reverse variable, comparison, data interpretation, etc.)
while locking hard formulas and pedagogical objectives.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from packages.schemas.models import VariationBlueprint, SeedQuestionAnalysis, RAGKnowledgeContext
from apps.generator.blueprint import PHYSICS_KINEMATICS_SCENARIOS

STRATEGIES = [
    "direct_calculation",
    "real_world_scenario",
    "reverse_variable",
    "comparison",
    "data_interpretation",
    "unit_conversion",
    "constraint_scenario"
]


class VariationPlanner(ABC):
    """Abstract interface for planning question variation strategies."""

    @abstractmethod
    def plan_variations(
        self,
        analysis: SeedQuestionAnalysis,
        count: int,
        rag_context: Optional[RAGKnowledgeContext] = None,
        start_index: int = 0
    ) -> List[VariationBlueprint]:
        """Generate N diverse VariationBlueprint objects adhering to seed analysis."""
        pass

    @abstractmethod
    def repair_blueprint(
        self,
        blueprint: VariationBlueprint,
        failure_reasons: List[str]
    ) -> VariationBlueprint:
        """Repair a blueprint based on specific evaluator failure reasons."""
        pass


class DefaultVariationPlanner(VariationPlanner):
    """Default Variation Planner producing diverse blueprint strategies and failure repairs."""

    def plan_variations(
        self,
        analysis: SeedQuestionAnalysis,
        count: int,
        rag_context: Optional[RAGKnowledgeContext] = None,
        start_index: int = 0
    ) -> List[VariationBlueprint]:
        blueprints: List[VariationBlueprint] = []

        for offset in range(count):
            index = start_index + offset
            strategy = STRATEGIES[index % len(STRATEGIES)]
            scen_tuple = PHYSICS_KINEMATICS_SCENARIOS[index % len(PHYSICS_KINEMATICS_SCENARIOS)]
            base_scen, base_dist, base_time = scen_tuple

            extra_entities = [
                "peloton rider", "bullet train", "sprint athlete", "freight truck", "speed boat",
                "cargo plane", "track athlete", "superbike", "competition bobsled", "quadcopter drone",
                "bathyscaphe", "electric streetcar", "sports car", "BASE jumper", "track runner",
                "funicular railway", "racing hydroplane", "solar vehicle", "orbital satellite", "artillery shell"
            ]
            if index >= len(PHYSICS_KINEMATICS_SCENARIOS):
                dist_val = round(base_dist + (index * 17.5), 1)
                time_val = round(base_time + (index % 7) + 1.0, 1)
                extra_idx = (index - len(PHYSICS_KINEMATICS_SCENARIOS)) % len(extra_entities)
                scenario_name = extra_entities[extra_idx]
            else:
                dist_val = base_dist
                time_val = base_time
                scenario_name = base_scen

            # Extract formula from analysis
            formula = analysis.formulas[0] if analysis.formulas else "v = d / t"

            bp = VariationBlueprint(
                domain=analysis.domain,
                topic=analysis.topic,
                learning_objective=analysis.learning_objective,
                formula=formula,
                task_type=analysis.question_type or "quantitative calculation",
                scenario=scenario_name,
                target_variable="velocity",
                known_variables={
                    "distance": dist_val,
                    "time": time_val
                },
                units={
                    "distance": "m",
                    "time": "s",
                    "velocity": "m/s"
                },
                strategy_name=strategy,
                target_bloom_level=analysis.bloom_level or "Apply"
            )
            blueprints.append(bp)

        return blueprints

    def repair_blueprint(
        self,
        blueprint: VariationBlueprint,
        failure_reasons: List[str]
    ) -> VariationBlueprint:
        """Analyze failure feedback and return a repaired VariationBlueprint."""
        repaired = blueprint.model_copy(deep=True)
        reasons_str = " ".join(failure_reasons).lower()

        # 1. Duplicate failure -> Change scenario and strategy pattern
        if "duplicate" in reasons_str:
            curr_strat_idx = STRATEGIES.index(repaired.strategy_name) if repaired.strategy_name in STRATEGIES else 0
            new_strat = STRATEGIES[(curr_strat_idx + 2) % len(STRATEGIES)]
            repaired.strategy_name = new_strat
            repaired.scenario = f"express {repaired.scenario}"
            if "distance" in repaired.known_variables:
                repaired.known_variables["distance"] += 120.0
            if "time" in repaired.known_variables:
                repaired.known_variables["time"] += 2.0

        # 2. Objective failure -> Lock/restore core learning objective
        if "objective" in reasons_str:
            repaired.target_bloom_level = "Apply"
            repaired.task_type = "quantitative calculation"

        # 3. Bloom taxonomy failure -> Aligns target Bloom level
        if "bloom" in reasons_str:
            repaired.target_bloom_level = "Apply"

        # 4. Difficulty failure -> Adjust numerical complexity/scale
        if "difficulty" in reasons_str:
            if "distance" in repaired.known_variables:
                repaired.known_variables["distance"] = round(repaired.known_variables["distance"] * 1.5, 1)

        # 5. Answer invalid -> Ensure valid clean numeric values
        if "answer" in reasons_str:
            repaired.known_variables["distance"] = 200.0
            repaired.known_variables["time"] = 10.0

        return repaired

