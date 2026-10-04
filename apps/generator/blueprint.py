"""
Variation Blueprint Generator module.
Creates structured blueprints with locked formulas, controlled numeric parameters,
and rich entity/scenario diversity prior to LLM natural language synthesis.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from packages.schemas.models import VariationBlueprint
from apps.generator.interfaces import ParsedQuestion, LearningObjective

# Diverse real-world scenarios for physics kinematics variations (60 distinct scenarios)
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
    ("satellite", 700.0, 14.0),
    ("projectile", 540.0, 18.0),
    ("spring-mass vehicle", 150.0, 5.0),
    ("high-speed train", 900.0, 30.0),
    ("charged particle", 80.0, 4.0),
    ("velodrome cyclist", 330.0, 11.0),
    ("banked automobile", 480.0, 16.0),
    ("falling raindrop", 90.0, 6.0),
    ("circular motion particle", 210.0, 7.0),
    ("space rocket", 1200.0, 40.0),
    ("roller coaster cart", 260.0, 13.0),
    ("marathon athlete", 380.0, 19.0),
    ("freefall skydiver", 650.0, 13.0),
    ("mountain cable car", 180.0, 12.0),
    ("ice track bobsled", 270.0, 9.0),
    ("recurve bow arrow", 140.0, 2.0),
    ("ferris wheel cabin", 100.0, 20.0),
    ("ocean submersible", 220.0, 11.0),
    ("warehouse crate", 75.0, 5.0),
    ("drag strip motorcycle", 400.0, 8.0),
    ("space probe", 1500.0, 50.0),
    ("outfield baseball", 120.0, 4.0),
    ("bouncing rubber ball", 50.0, 2.0),
    ("ballistic pendulum", 60.0, 3.0),
    ("ice hockey puck", 160.0, 4.0),
    ("steep ramp bobsled", 310.0, 10.0),
    ("hovering drone", 130.0, 10.0),
    ("building elevator", 85.0, 5.0),
    ("catapult stone", 350.0, 7.0),
    ("ski jumper", 225.0, 9.0),
    ("lake hydroplane", 490.0, 14.0),
    ("particle accelerator", 110.0, 2.0),
    ("billiard ball", 45.0, 3.0),
    ("fort cannonball", 420.0, 12.0),
    ("pendulum clock bob", 36.0, 2.0),
    ("industrial conveyor", 185.0, 37.0),
    ("transit station tram", 290.0, 10.0),
    ("high-jump athlete", 64.0, 8.0),
    ("rolling boulder", 175.0, 7.0),
    ("supersonic jet", 1600.0, 40.0),
    ("maglev train", 1050.0, 21.0),
    ("tacking sailboat", 230.0, 23.0)
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

