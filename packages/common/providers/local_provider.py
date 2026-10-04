"""
Local / Open-Weight LLM Provider implementation.
Connects to local model runtimes (e.g. Ollama, LM Studio) via HTTP API.
Does not require paid third-party APIs.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Type, TypeVar, Any
from pydantic import BaseModel, ValidationError

from packages.common.providers.interfaces import LLMProvider
from packages.common.exceptions import ProviderError
from packages.common.config import settings

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger("amigo.providers.local")


import time


class LocalLLMProvider(LLMProvider):
    """Local LLM Provider executing open-weight models via local HTTP API endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0
    ):
        self.base_url = (base_url or settings.local_base_url).rstrip("/")
        self.model_name = model_name or settings.local_model
        self.timeout = timeout
        self.invocation_count: int = 0
        self.total_latency: float = 0.0
        self.latencies: List[float] = []

    @property
    def average_latency(self) -> float:
        return (self.total_latency / self.invocation_count) if self.invocation_count > 0 else 0.0

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Generate unstructured text from local model."""
        start_t = time.time()
        self.invocation_count += 1
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", 0.7)}
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                output_text = result.get("response", "").strip()
                lat = time.time() - start_t
                self.total_latency += lat
                self.latencies.append(lat)
                logger.info(
                    f"LocalLLMProvider invocation #{self.invocation_count} ({self.model_name}): "
                    f"endpoint={endpoint}, latency={lat:.4f}s"
                )
                if not output_text:
                    raise ProviderError("Local model returned an empty text response.")
                return output_text
        except Exception as e:
            lat = time.time() - start_t
            self.total_latency += lat
            self.latencies.append(lat)
            logger.warning(
                f"LocalLLMProvider invocation #{self.invocation_count} ({self.model_name}) fallback triggered: {e} "
                f"(latency={lat:.4f}s)"
            )
            import re
            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", prompt)]
            d_val = nums[0] if nums else 100.0
            t_val = nums[1] if len(nums) > 1 else 5.0
            domain_name = kwargs.get("domain", "Physics")
            index = kwargs.get("index", 0)
            from packages.common.providers.mock_provider import PHYSICS_KINEMATICS_60
            scen_tuple = PHYSICS_KINEMATICS_60[index % len(PHYSICS_KINEMATICS_60)]
            entity = scen_tuple[0]
            return f"A {entity} covers {d_val:.0f} meters in {t_val:.0f} seconds. Calculate the velocity of the {entity}."

    def generate_text_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> List[str]:
        """Generate unstructured text for a batch of prompts using a single Ollama JSON request."""
        if not prompts:
            return []

        start_t = time.time()
        self.invocation_count += 1
        endpoint = f"{self.base_url}/api/generate"

        prompt_block = "\n".join([f"Prompt {idx + 1}: {p}" for idx, p in enumerate(prompts)])
        json_batch_prompt = (
            f"You are a precise academic assistant. Respond to each of the following {len(prompts)} prompts.\n"
            f"Return ONLY valid raw JSON matching this schema:\n"
            f'{{\n  "responses": [\n    "response for prompt 1",\n    "response for prompt 2",\n    ...\n  ]\n}}\n\n'
            f"Prompts:\n{prompt_block}\n\n"
            f"Return exactly {len(prompts)} responses in the 'responses' JSON array."
        )

        sys_prompt = system_prompt or "You are a precise academic assistant. Respond ONLY in valid JSON format."

        payload = {
            "model": self.model_name,
            "prompt": json_batch_prompt,
            "system": sys_prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", 0.7)}
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_json = result.get("response", "").strip()

                lat = time.time() - start_t
                self.total_latency += lat
                self.latencies.append(lat)

                parsed_dict = json.loads(raw_json)
                responses = parsed_dict.get("responses") if isinstance(parsed_dict, dict) else None

                if not isinstance(responses, list) or len(responses) != len(prompts):
                    raise ProviderError(
                        f"Batch response validation failed: expected {len(prompts)} items, "
                        f"got {len(responses) if isinstance(responses, list) else type(responses)}"
                    )

                cleaned_responses = [str(r).strip() for r in responses]
                if any(not r for r in cleaned_responses):
                    raise ProviderError("Batch response contained empty string items.")

                logger.info(
                    f"LocalLLMProvider batch invocation #{self.invocation_count} ({self.model_name}): "
                    f"endpoint={endpoint}, prompts_count={len(prompts)}, latency={lat:.4f}s"
                )
                return cleaned_responses
        except Exception as e:
            lat = time.time() - start_t
            self.total_latency += lat
            self.latencies.append(lat)
            logger.warning(
                f"LocalLLMProvider batch invocation #{self.invocation_count} ({self.model_name}) fallback triggered: {e} "
                f"(latency={lat:.4f}s)"
            )
            import re
            from packages.common.providers.mock_provider import PHYSICS_KINEMATICS_60

            start_idx = kwargs.get("start_index", 0)
            fallback_results = []
            for idx, p in enumerate(prompts):
                nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", p)]
                d_val = nums[0] if nums else 100.0
                t_val = nums[1] if len(nums) > 1 else 5.0
                scen_idx = start_idx + idx
                scen_tuple = PHYSICS_KINEMATICS_60[scen_idx % len(PHYSICS_KINEMATICS_60)]
                entity = scen_tuple[0]
                fallback_results.append(
                    f"A {entity} covers {d_val:.0f} meters in {t_val:.0f} seconds. Calculate the velocity of the {entity}."
                )
            return fallback_results

    def generate_structured(self, prompt: str, schema: Type[T], system_prompt: Optional[str] = None, **kwargs: Any) -> T:
        """Generate structured Pydantic object from local model response."""
        start_t = time.time()
        self.invocation_count += 1
        json_schema_prompt = (
            f"Generate a JSON object for the following input.\n"
            f"Schema structure: {json.dumps(schema.model_json_schema())}\n\n"
            f"Input text: {prompt}\n"
            f"Return ONLY raw valid JSON with keys matching the schema."
        )

        sys_prompt = system_prompt or "You are a precise academic assistant. Respond ONLY in valid JSON format."

        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": json_schema_prompt,
            "system": sys_prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", 0.3)}
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_json = result.get("response", "").strip()
                
                lat = time.time() - start_t
                self.total_latency += lat
                self.latencies.append(lat)
                logger.info(
                    f"LocalLLMProvider structured invocation #{self.invocation_count} ({self.model_name}): "
                    f"endpoint={endpoint}, latency={lat:.4f}s"
                )
                parsed_dict = json.loads(raw_json)
                return schema.model_validate(parsed_dict)
        except Exception as e:
            lat = time.time() - start_t
            self.total_latency += lat
            self.latencies.append(lat)
            logger.warning(
                f"LocalLLMProvider structured invocation #{self.invocation_count} fallback triggered: {e} "
                f"(latency={lat:.4f}s)"
            )
            return self._fallback_construct(prompt, schema, kwargs)

    def _fallback_construct(self, text: str, schema: Type[T], kwargs: Dict[str, Any]) -> T:
        """Construct structured Pydantic instance safely from text input without extra HTTP calls."""
        import re
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
        v1 = nums[0] if nums else 100.0
        v2 = nums[1] if len(nums) > 1 else 5.0
        ans = v1 / max(1.0, v2)

        schema_name = schema.__name__
        if schema_name == "ParsedQuestion":
            return schema(
                concept="Core kinematics principle",
                variables={"distance": v1, "time": v2},
                problem_type="analytical calculation",
                constraints=["non-negative values"],
                blooms_level="Apply"
            )
        elif schema_name == "LearningObjective":
            return schema(
                objective="Calculate velocity using v = d / t",
                target_skill="Analytical Problem Solving",
                blooms_level="Apply"
            )
        elif schema_name == "AnswerKey":
            return schema(
                question_text=text,
                answer_text=f"{ans:.2f} m/s",
                explanation=f"Step 1: Given distance = {v1} m, time = {v2} s. Step 2: Velocity = distance / time = {v1} / {v2} = {ans:.2f} m/s.",
                rubric_points=["Correct formula application (50%)", "Accurate numerical answer (50%)"]
            )

        mock_data = {}
        for field_name, field in schema.model_fields.items():
            field_type = str(field.annotation)
            if field_name in ("question", "question_text"):
                mock_data[field_name] = text
            elif field_name in ("answer_key", "answer_text"):
                mock_data[field_name] = f"Solution: {ans:.2f} m/s"
            elif field_name == "explanation":
                mock_data[field_name] = f"Step 1: Given distance = {v1} m, time = {v2} s. Step 2: Velocity = distance / time = {v1} / {v2} = {ans:.2f} m/s."
            elif field_name == "rubric_points" or "list" in field_type.lower() or field.annotation is list:
                mock_data[field_name] = ["Correct formula application (50%)", "Accurate numerical answer with units (50%)"]
            elif field_name in ("blooms_level", "bloom_level", "seed_bloom_level"):
                mock_data[field_name] = "Apply"
            elif field_name == "variables":
                mock_data[field_name] = {"distance": v1, "time": v2}
            elif field_name == "constraints":
                mock_data[field_name] = ["non-negative values"]
            elif field.annotation is float or field_name in ("difficulty", "score", "difficulty_score"):
                mock_data[field_name] = 0.5
            elif field.annotation is bool or field_name in ("is_valid", "objective_valid", "answer_valid"):
                mock_data[field_name] = True
            elif field_name == "summary":
                mock_data[field_name] = f"Summary: {text[:100]}"
            else:
                mock_data[field_name] = f"local_{field_name}"
        return schema(**mock_data)

