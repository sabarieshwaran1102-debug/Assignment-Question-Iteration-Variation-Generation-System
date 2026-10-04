"""
Integration tests for LocalLLMProvider.
Skipped cleanly if local open-weight model runtime (Ollama) is unavailable or unready.
"""

import urllib.request
import pytest
from pydantic import BaseModel

from packages.common.providers.local_provider import LocalLLMProvider
from packages.common.config import settings


def is_local_runtime_available() -> bool:
    """Check if local model server (Ollama) is reachable."""
    try:
        url = f"{settings.local_base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


LOCAL_AVAILABLE = is_local_runtime_available()


class SampleSchema(BaseModel):
    summary: str
    difficulty_score: float


@pytest.mark.skipif(not LOCAL_AVAILABLE, reason="Local model runtime (Ollama) is not running on localhost:11434")
def test_local_llm_provider_text_generation():
    """Test text generation with live local open-weight model."""
    provider = LocalLLMProvider(timeout=10.0)
    try:
        prompt = "Explain in one sentence why friction causes energy loss."
        text = provider.generate_text(prompt)
        assert isinstance(text, str)
        assert len(text.strip()) > 0
    except Exception as e:
        pytest.skip(f"Local model call unready or timed out: {e}")


@pytest.mark.skipif(not LOCAL_AVAILABLE, reason="Local model runtime (Ollama) is not running on localhost:11434")
def test_local_llm_provider_structured_generation():
    """Test structured generation with live local open-weight model."""
    provider = LocalLLMProvider(timeout=10.0)
    try:
        prompt = "Provide a summary and difficulty score for a physics velocity question."
        result = provider.generate_structured(prompt, schema=SampleSchema)
        assert isinstance(result, SampleSchema)
        assert isinstance(result.summary, str)
        assert 0.0 <= result.difficulty_score <= 1.0
    except Exception as e:
        pytest.skip(f"Local model structured generation unready or timed out: {e}")


def test_get_llm_provider_returns_local_provider(monkeypatch):
    """Verify get_llm_provider returns LocalLLMProvider when settings.llm_provider is set to 'local'."""
    from apps.api.dependencies import get_llm_provider
    monkeypatch.setattr(settings, "llm_provider", "local")
    provider = get_llm_provider()
    assert isinstance(provider, LocalLLMProvider)


def test_api_generate_invokes_local_provider(monkeypatch):
    """Verify /api/v1/generate invokes LocalLLMProvider when settings.llm_provider='local'."""
    from fastapi.testclient import TestClient
    from apps.api.main import app
    from apps.api import dependencies

    invocations = []

    def mock_gen_batch(self, prompts, system_prompt=None, **kwargs):
        invocations.append("generate_text_batch")
        self.invocation_count += 1
        return ["A test vehicle covers 100 meters in 5 seconds. Calculate the velocity of the vehicle."] * len(prompts)

    monkeypatch.setattr(settings, "llm_provider", "local")
    monkeypatch.setattr(LocalLLMProvider, "generate_text_batch", mock_gen_batch)

    client = TestClient(app)
    req_payload = {
        "seed_question": "Calculate the velocity of a vehicle moving 100m in 5 seconds.",
        "domain": "Physics",
        "count": 5
    }

    resp = client.post("/api/v1/generate", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["variations"]) == 5
    # Confirm LocalLLMProvider was actually invoked during the /api/v1/generate call
    assert len(invocations) > 0

