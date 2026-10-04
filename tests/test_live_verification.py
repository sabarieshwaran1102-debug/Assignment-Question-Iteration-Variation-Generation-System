"""
Live verification test for Phase 1 endpoints, data structures, and edge cases.
"""

from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)


def test_phase1_live_verification_full():
    # 1. Test GET /api/v1/health
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["status"] == "healthy"
    assert "version" in health_data
    assert "phase" in health_data

    # 2. Test GET /api/v1/domains (verify at least 5 selectable domains)
    res_domains = client.get("/api/v1/domains")
    assert res_domains.status_code == 200
    domains_data = res_domains.json()
    assert "domains" in domains_data
    domains_list = domains_data["domains"]
    assert len(domains_list) >= 5
    for domain_item in domains_list:
        assert "id" in domain_item
        assert "name" in domain_item
        assert "description" in domain_item

    # 3. Test POST /api/v1/generate with specific prompt request
    req_payload = {
        "seed_question": "Calculate the velocity of a vehicle moving 100m in 5 seconds.",
        "domain": "Physics",
        "count": 10
    }
    res_gen = client.post("/api/v1/generate", json=req_payload)
    assert res_gen.status_code == 200
    data = res_gen.json()

    # Verify official PS8 response fields
    assert "variations" in data
    assert "duplicate_rate" in data
    assert isinstance(data["duplicate_rate"], float)

    variations = data["variations"]
    assert len(variations) == 10

    questions_set = set()
    for var in variations:
        assert "question" in var
        assert "answer_key" in var
        assert "difficulty" in var

        assert isinstance(var["question"], str) and var["question"].strip() != ""
        assert isinstance(var["answer_key"], str) and var["answer_key"].strip() != ""
        assert isinstance(var["difficulty"], float) and 0.0 <= var["difficulty"] <= 1.0

        questions_set.add(var["question"])

    # Verify that generated questions are NOT identical copies
    assert len(questions_set) == 10, f"Expected 10 unique questions, got {len(questions_set)}"

    # 4. Test invalid count values
    res_inv_count = client.post("/api/v1/generate", json={
        "seed_question": "Calculate velocity",
        "domain": "Physics",
        "count": -5
    })
    assert res_inv_count.status_code == 422

    # 5. Test empty seed question
    res_inv_seed = client.post("/api/v1/generate", json={
        "seed_question": "",
        "domain": "Physics",
        "count": 10
    })
    assert res_inv_seed.status_code == 400
