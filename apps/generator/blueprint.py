"""
Variation Blueprint Generator module.
Creates structured blueprints with locked formulas, controlled numeric parameters,
and rich entity/scenario diversity prior to LLM natural language synthesis.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from packages.schemas.models import VariationBlueprint
from apps.generator.interfaces import ParsedQuestion, LearningObjective

# Diverse real-world scenarios for physics kinematics variations
PHYSICS_KINEMATICS_SCENARIOS = [
    ("cyclist", 240.0, 12.0),
    ("train", 450.0, 15.0),
    ("runner", 180.0, 9.0),
    ("delivery truck", 300.0, 15.0),
    ("boat", 120.0, 6.0),
    ("aircraft", 600.0, 20.0),
    ("athlete", 100.0, 5.0),
    ("motorcycle", 360.0, 12.0),
    ("bobsled", 250.0, 10.0),
    ("drone", 150.0, 10.0),
    ("submersible", 200.0, 8.0),
    ("tram car", 280.0, 14.0),
    ("car", 400.0, 20.0),
    ("skydiver", 500.0, 25.0),
    ("marathon runner", 320.0, 16.0),
    ("cable car", 160.0, 8.0),
    ("hydroplane", 420.0, 14.0),
    ("solar car", 350.0, 10.0),
]


class BlueprintGenerator(ABC):
    """Abstract interface for generating structured variation blueprints."""

    @abstractmethod
    def generate_blueprints(
        self,
        parsed: ParsedQuestion,
        objective: LearningObjective,
        count: int,
        start_index: int = 0
    ) -> List[VariationBlueprint]:
        """Generate N variation blueprints with hard constraints and controlled parameters."""
        pass


class DefaultBlueprintGenerator(BlueprintGenerator):
    """Default Blueprint Generator for quantitative and conceptual variation pipelines."""

    def generate_blueprints(
        self,
        parsed: Optional[ParsedQuestion],
        objective: Optional[LearningObjective],
        count: int,
        start_index: int = 0
    ) -> List[VariationBlueprint]:
        blueprints: List[VariationBlueprint] = []

        domain = parsed.domain if parsed else "Physics"
        topic = getattr(parsed, "concept", "Kinematics")
        obj_text = objective.objective if objective else "Calculate velocity using distance and time"

        for offset in range(count):
            index = start_index + offset
            scen_tuple = PHYSICS_KINEMATICS_SCENARIOS[index % len(PHYSICS_KINEMATICS_SCENARIOS)]
            scenario_name, dist_val, time_val = scen_tuple

            # If parsed contains custom variables, use them if available
            if parsed and parsed.variables and "v_1" in parsed.variables and "v_2" in parsed.variables:
                pass

            bp = VariationBlueprint(
                domain=domain,
                topic=topic,
                learning_objective=obj_text,
                formula="v = d / t",
                task_type="quantitative calculation",
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
                }
            )
            blueprints.append(bp)

        return blueprints

