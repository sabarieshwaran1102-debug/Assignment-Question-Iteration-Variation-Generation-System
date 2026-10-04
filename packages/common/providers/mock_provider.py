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
    ("cyclist traveling on a velodrome", "travel distance", "average velocity"),
    ("high-speed train navigating a rail route", "route distance", "average velocity"),
    ("marathon runner pacing a course", "course distance", "average velocity"),
    ("delivery truck moving along a route", "transit distance", "average speed"),
    ("motorboat cruising across a lake", "lake distance", "average velocity"),
    ("aircraft flying a straight corridor", "corridor distance", "average speed"),
    ("submersible traveling in ocean water", "descent distance", "average descent velocity"),
    ("cargo crate sliding on a conveyor", "conveyor distance", "average speed"),
    ("motorcycle traveling on a highway", "highway distance", "average speed"),
    ("space probe cruising through deep space", "probe distance", "average speed"),
    ("bobsled moving along an ice track", "track distance", "average velocity"),
    ("arrow flying along a target range", "range distance", "average flight velocity"),
    ("ferris wheel cabin sweeping an arc", "arc distance", "average speed"),
    ("drone cruising along a flight path", "path distance", "average speed"),
    ("elevator ascending a shaft", "shaft distance", "average velocity"),
    ("ski jumper sliding down an inrun", "inrun distance", "average speed"),
    ("hydroplane boat skimming a river", "river distance", "average cruising speed"),
    ("billiard ball rolling across a table", "table distance", "average velocity"),
    ("stone launched along a trajectory", "flight distance", "average velocity"),
    ("pendulum bob swinging through an arc", "arc distance", "average speed"),
    ("tram car moving between stations", "station distance", "average speed"),
    ("high-jump athlete sprinting the approach", "approach distance", "average speed"),
    ("supersonic jet flying a test route", "route distance", "average velocity"),
    ("maglev train moving along a guide track", "track distance", "average velocity"),
    ("sailboat sailing across a bay", "bay distance", "average speed"),
    ("disc golfer throwing a frisbee disc", "flight distance", "average flight speed"),
    ("tetherball orbiting a central pole", "orbit distance", "average speed"),
    ("bungee jumper falling in freefall", "fall distance", "average speed"),
    ("water jet streaming from a hose", "stream distance", "average exit velocity"),
    ("curling stone sliding over ice", "ice distance", "average speed"),
    ("satellite orbiting Earth", "orbital path distance", "average orbital speed"),
    ("paraglider gliding along a ridge", "ridge distance", "average speed"),
    ("acoustic wave traveling in water", "sound distance", "average wave velocity"),
    ("electron moving in a cathode ray tube", "tube distance", "average speed"),
    ("wind turbine tip sweeping an arc", "tip arc distance", "average speed"),
    ("slingshot projectile flying downrange", "downrange distance", "average launch speed"),
    ("glider aircraft coasting in still air", "gliding distance", "average glide velocity"),
    ("freight train moving across level track", "track distance", "average speed"),
    ("lunar lander descending to moon surface", "descent distance", "average descent velocity"),
    ("solar sail probe traveling near Mars", "probe path distance", "average speed"),
    ("racing car navigating a straightaway", "straightaway distance", "average velocity"),
    ("robot rover moving across desert sand", "rover distance", "average speed"),
    ("hovercraft skimming over marshland", "marsh distance", "average speed"),
    ("ambulance traveling along an expressway", "expressway distance", "average speed"),
    ("cargo ship crossing an ocean channel", "channel distance", "average cruising velocity"),
    ("zipline rider descending a canyon", "canyon distance", "average descent speed"),
    ("cable car moving up a mountain peak", "peak distance", "average speed"),
    ("snowmobile cruising over a frozen tundra", "tundra distance", "average speed"),
    ("patrol boat navigating a coastal bay", "coastal distance", "average speed"),
    ("hyperloop pod traveling in a vacuum tube", "tube distance", "average speed"),
    ("kayaker paddling down a river rapid", "river distance", "average speed"),
    ("scooter traveling through city streets", "street distance", "average speed"),
    ("monorail train proceeding along an elevated track", "elevated track distance", "average speed"),
    ("roller coaster cart coasting down a hill", "hill distance", "average speed"),
    ("bicycles sharing a suburban bike path", "path distance", "average speed"),
    ("solar-powered car driving across a desert", "desert distance", "average speed"),
    ("airship floating over a city stadium", "stadium distance", "average speed"),
    ("electric bus commuting between transit stops", "transit distance", "average speed"),
    ("skateboarder rolling down a paved ramp", "ramp distance", "average speed"),
    ("ferry boat crossing a wide river delta", "delta distance", "average speed"),
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

        stop_words = {
            "a", "an", "the", "in", "on", "at", "of", "to", "for", "is", "are",
            "and", "or", "meters", "seconds", "calculate", "what", "find", "given",
            "determine", "compute", "resulting", "constant", "average", "its"
        }
        words = [w for w in re.findall(r"\w+", text_clean) if w not in stop_words and len(w) > 2]
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
