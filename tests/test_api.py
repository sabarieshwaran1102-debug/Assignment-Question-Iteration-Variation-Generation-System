"""
API Endpoint Unit Tests using FastAPI TestClient.
Tests official PS8 contract for /generate, /domains, /health.
"""

from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "phase" in data


def test_domains_endpoint():
    response = client.get("/api/v1/domains")
    assert response.status_code == 200
    data = response.json()
    assert "domains" in data
    domains = data["domains"]
    # PS8 requires at least 5 selectable domains
    assert len(domains) >= 5
    domain_names = [d["name"] for d in domains]
    assert "Computer Science" in domain_names
    assert "Physics" in domain_names
    assert "Mathematics" in domain_names


def test_generate_endpoint_official_ps8_contract():
    payload = {
        "seed_question": "Calculate velocity when distance is 100m and time is 5 seconds.",
        "domain": "Physics",
        "count": 60
    }

    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify official PS8 response structure:
    # { "variations": [ { "question": "...", "answer_key": "...", "difficulty": 0.0 } ], "duplicate_rate": 0.0 }
    assert "variations" in data
    assert "duplicate_rate" in data
    assert len(data["variations"]) == 60
    assert isinstance(data["duplicate_rate"], float)

    first_item = data["variations"][0]
    assert "question" in first_item
    assert "answer_key" in first_item
    assert "difficulty" in first_item
    assert isinstance(first_item["difficulty"], float)


def test_generate_endpoint_invalid_request():
    # Test empty seed question
    payload = {
        "seed_question": "",
        "domain": "Physics",
        "count": 60
    }
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 400

    # Test invalid count constraint
    payload_invalid_count = {
        "seed_question": "Valid seed question",
        "domain": "Physics",
        "count": -10
    }
    response_ic = client.post("/api/v1/generate", json=payload_invalid_count)
    assert response_ic.status_code == 422
