"""
Deterministic Mock Provider implementations for offline development and testing.
Produces non-repetitive, deterministic variations without external API dependencies.
"""

import hashlib
import re
from typing import Dict, List, Optional, Type, TypeVar, Any
import numpy as np
from pydantic import BaseModel

from packages.common.providers.interfaces import LLMProvider, EmbeddingProvider, VectorStore

T = TypeVar("T", bound=BaseModel)

# 60 distinct kinematics scenarios for index 0..59
PHYSICS_KINEMATICS_60 = [
    ("mass sliding on an inclined plane", "incline friction coefficient", "stopping distance and net force"),
    ("satellite in circular Earth orbit", "orbital radius", "gravitational velocity and orbital period"),
    ("projectile launched from an elevated cliff", "launch velocity", "maximum trajectory altitude and flight time"),
    ("spring-mass system on a horizontal track", "spring constant", "oscillation period and max speed"),
    ("high-speed train braking on a rail grade", "deceleration rate", "stopping distance and work done"),
    ("charged electron moving through a magnetic field", "magnetic flux density", "cyclotron radius and velocity"),
    ("cyclist accelerating along a straight velodrome", "pedal force", "final speed and distance covered"),
    ("automobile rounding a banked circular curve", "bank angle", "maximum safe velocity without skidding"),
    ("falling raindrop reaching terminal velocity", "air resistance coefficient", "terminal velocity and acceleration"),
    ("particle in uniform circular motion", "rotation radius", "centripetal acceleration and speed"),
    ("rocket accelerating during first-stage burn", "thrust force", "velocity reached after 10 seconds"),
    ("roller coaster cart at the loop crest", "loop radius", "minimum entry velocity to maintain contact"),
    ("marathon runner maintaining constant pace", "stride rate", "total distance covered in duration"),
    ("skydiver before deploying parachute", "drag coefficient", "downward velocity at 5 seconds"),
    ("elevated cable car ascending a mountain", "cable tension", "vertical velocity and time required"),
    ("bobsled navigating an ice track turn", "track curvature", "centripetal force and velocity"),
    ("arrow released from a recurve bow", "draw force", "initial launch velocity and flight range"),
    ("ferris wheel rotating at steady speed", "wheel radius", "tangential speed of passenger cabin"),
    ("submersible descending in ocean water", "buoyant imbalance", "downward terminal descent speed"),
    ("cargo crate sliding across a warehouse floor", "pushing force", "acceleration and distance moved"),
    ("motorcycle accelerating on a drag strip", "engine torque", "elapsed time to reach 100 km/h"),
    ("space probe performing a planetary flyby", "closest approach distance", "deflection angle and final velocity"),
    ("baseball hit into outfield trajectory", "bat exit speed", "hang time and horizontal distance"),
    ("bouncing rubber ball returning upward", "coefficient of restitution", "rebound height and speed"),
    ("bullet fired into a ballistic pendulum", "block mass", "post-collision pendulum height and velocity"),
    ("hockey puck gliding over smooth ice", "ice friction", "distance traveled before stopping"),
    ("bobsled accelerating down a steep ramp", "ramp slope", "speed at the bottom of the ramp"),
    ("drone hovering in gusty wind conditions", "rotor thrust", "corrective velocity and position stability"),
    ("elevator descending between building floors", "cable deceleration", "stopping time and distance"),
    ("cannonball launched at a fort wall", "elevation angle", "impact velocity and flight duration"),
    ("ski jumper descending the takeoff inrun", "inrun height", "takeoff speed at the lip"),
    ("hydroplane boat skimming over lake water", "water drag", "maximum steady cruising velocity"),
    ("particle accelerated in a linear accelerator", "electric field", "final relativistic kinetic speed"),
    ("billiard ball colliding elastically with a cushion", "impact angle", "rebound angle and velocity"),
    ("catapult launching a stone projectile", "arm tension", "launch velocity and horizontal range"),
    ("pendulum clock bob swinging through equilibrium", "rod length", "maximum linear velocity at lowest point"),
    ("conveyor belt carrying industrial packages", "belt drive speed", "transit time across factory floor"),
    ("tram car accelerating from a transit station", "motor power", "time to reach maximum cruise speed"),
    ("high-jump athlete launching from the track", "takeoff angle", "peak vertical height cleared"),
    ("archaeological boulder rolling down a hill", "hill slope", "angular and translational kinetic velocity"),
    ("supersonic jet breaking the sound barrier", "mach number", "flight speed and shockwave angle"),
    ("magnetic levitation train accelerating smoothly", "magnetic drive force", "acceleration and peak velocity"),
    ("sailboat tacking against a headwind", "wind speed", "effective forward boat velocity"),
    ("disc golfer throwing a driver disc", "release speed", "flight glide distance and turn rate"),
    ("tetherball orbiting around a central pole", "rope length", "rotation speed and rope tension"),
    ("bungee jumper falling before cord tension", "freefall height", "maximum velocity reached before tension"),
    ("water jet exiting a firehose nozzle", "nozzle pressure", "water exit velocity and vertical reach"),
    ("curling stone sliding down a sheet of ice", "sweeping friction", "distance to target house circle"),
    ("satellite performing an orbital plane change", "delta-v impulse", "new orbital velocity vector"),
    ("paraglider gliding in a thermal updraft", "sink rate", "forward air speed and glide distance"),
    ("acoustic pulse traveling through seabed sediment", "sediment density", "sound wave travel velocity"),
    ("electron beam in a cathode ray tube", "deflection voltage", "electron velocity and screen impact"),
    ("bobsled team pushing off at the start line", "pushing force", "initial speed entering the track"),
    ("wind turbine blade tip rotating in a gale", "rotor diameter", "linear velocity of the blade tip"),
    ("slingshot launching a steel bearing", "elastic stretch", "launch speed and kinetic energy"),
    ("glider aircraft descending in still air", "lift-to-drag ratio", "gliding velocity and touchdown time"),
    ("freight train coasting on level tracks", "rolling resistance", "distance coasted before coming to rest"),
    ("lunar lander descending to moon surface", "thruster deceleration", "touchdown velocity and fuel time"),
    ("solar sail spacecraft accelerating in deep space", "radiation pressure", "continuous velocity accumulation"),
    ("racing car negotiating an S-bend chicane", "tire traction", "maximum entry and exit speeds"),
]

CS_ALGORITHM_SCENARIOS = [
    ("array sorting algorithm execution", "array size N", "asymptotic comparison count"),
    ("binary search tree key lookup operation", "tree depth", "maximum search path comparisons"),
    ("lock-free concurrent queue pipeline", "thread concurrency", "contention latency"),
    ("Dijkstra shortest path graph traversal", "vertex count", "minimum path weight"),
    ("B-tree index node splitting process", "tree order", "maximum node capacity"),
]


class MockLLMProvider(LLMProvider):
    """Deterministic LLM Provider producing realistic, distinct variations without external LLMs."""

    def __init__(self, seed_offset: int = 0):
        self.seed_offset = seed_offset

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        domain = kwargs.get("domain", "Physics")
        index = kwargs.get("index", 0)
        return self._generate_deterministic_variation_text(prompt, domain, index)

    def generate_structured(self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None, **kwargs: Any) -> T:
        domain = kwargs.get("domain", "Physics")
        index = kwargs.get("index", 0)
        
        schema_name = schema.__name__
        
        if schema_name == "ParsedQuestion":
            return schema(
                concept="Core domain principle",
                variables={"param_a": 10, "param_b": 5},
                problem_type="analytical calculation",
                constraints=["non-negative values", "steady state"],
                blooms_level="Apply"
            )
        elif schema_name == "LearningObjective":
            return schema(
                objective=f"Evaluate principles of {domain} in practical problem-solving scenarios.",
                target_skill="Analytical Problem Solving",
                blooms_level="Apply"
            )
        elif schema_name == "AnswerKey":
            q_text = self._generate_deterministic_variation_text(prompt, domain, index)
            ans_val = (index + 1) * 12.5
            return schema(
                question_text=q_text,
                answer_text=f"Solution: {ans_val:.2f}",
                explanation=f"Step 1: Identify given parameters. Step 2: Apply core relationship for {domain}. Result = {ans_val:.2f}.",
                rubric_points=["Correct setup (40%)", "Accurate calculation (40%)", "Correct units (20%)"]
            )
        else:
            mock_data = {}
            for field_name, field in schema.model_fields.items():
                if field_name == "question":
                    mock_data[field_name] = self._generate_deterministic_variation_text(prompt, domain, index)
                elif field_name in ("answer_key", "answer_text"):
                    mock_data[field_name] = f"Detailed solution for variation {index + 1}: Value = {(index + 1) * 15:.2f}"
                elif field_name in ("difficulty", "score"):
                    mock_data[field_name] = round(0.3 + ((index * 7 + 3) % 50) / 100.0, 2)
                elif field_name == "is_valid":
                    mock_data[field_name] = True
                else:
                    mock_data[field_name] = f"mock_{field_name}_{index}"
            return schema(**mock_data)

    def _generate_deterministic_variation_text(self, prompt: str, domain: str, index: int) -> str:
        prompt_lower = prompt.lower()
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", prompt)]
        
        if domain == "Computer Science":
            scenarios = CS_ALGORITHM_SCENARIOS
        else:
            scenarios = PHYSICS_KINEMATICS_60

        scen_tuple = scenarios[index % len(scenarios)]
        topic, param_name, target_metric = scen_tuple
        
        base_num1 = numbers[0] if numbers else 100.0
        base_num2 = numbers[1] if len(numbers) > 1 else 5.0

        v1 = round(base_num1 * (1.0 + (index + 1) * 0.17), 2)
        v2 = round(base_num2 * (1.0 + (index + 1) * 0.23), 2)

        return (
            f"[Variation #{index + 1}] For {topic} where {param_name} = {v1:.2f} "
            f"and secondary parameter = {v2:.2f}, calculate the {target_metric}."
        )


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic Embedding Provider generating pseudo-embeddings via character n-gram hashing."""

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        text_clean = text.lower().strip()
        vec = np.zeros(self.dimension, dtype=np.float64)
        
        for i in range(len(text_clean) - 2):
            trigram = text_clean[i:i+3]
            h = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
            dim = h % self.dimension
            val = ((h >> 8) % 100) / 100.0 - 0.5
            vec[dim] += val

        words = re.findall(r"\w+", text_clean)
        for w in words:
            h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
            dim = h % self.dimension
            vec[dim] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class MockVectorStore(VectorStore):
    """In-memory vector storage and cosine similarity search engine."""

    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}

    def add(self, item_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        norm_v = np.array(vector, dtype=np.float64)
        norm = np.linalg.norm(norm_v)
        if norm > 0:
            norm_v = norm_v / norm
        self._items[item_id] = {
            "id": item_id,
            "vector": norm_v,
            "metadata": metadata or {}
        }

    def search(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._items:
            return []
            
        target_v = np.array(vector, dtype=np.float64)
        norm = np.linalg.norm(target_v)
        if norm > 0:
            target_v = target_v / norm

        results = []
        for item_id, item in self._items.items():
            sim = float(np.dot(target_v, item["vector"]))
            sim = max(0.0, min(1.0, sim))
            results.append({
                "id": item_id,
                "similarity": sim,
                "metadata": item["metadata"]
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self._items.clear()
