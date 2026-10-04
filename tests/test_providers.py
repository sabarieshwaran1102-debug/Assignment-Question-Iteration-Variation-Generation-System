"""
Unit tests for Provider interfaces and Mock implementation.
"""

from packages.common.providers.mock_provider import MockLLMProvider, MockEmbeddingProvider, MockVectorStore
from packages.schemas.models import AnswerKey


def test_mock_llm_provider_text_generation():
    provider = MockLLMProvider()
    q1 = provider.generate_text("Calculate velocity", domain="Physics", index=0)
    q2 = provider.generate_text("Calculate velocity", domain="Physics", index=1)
    
    assert q1 != q2
    assert "[Variation #1]" in q1
    assert "[Variation #2]" in q2


def test_mock_llm_provider_60_distinct_variations():
    provider = MockLLMProvider()
    generated = set()
    for i in range(60):
        var_text = provider.generate_text("Calculate speed of vehicle 100m in 5s", domain="Physics", index=i)
        generated.add(var_text)
    
    # Must produce 60 unique variations
    assert len(generated) == 60


def test_mock_llm_provider_structured_generation():
    provider = MockLLMProvider()
    ans = provider.generate_structured("Calculate speed", schema=AnswerKey, domain="Physics", index=0)
    assert isinstance(ans, AnswerKey)
    assert ans.question_text != ""
    assert ans.answer_text != ""


def test_mock_embedding_provider():
    provider = MockEmbeddingProvider(dimension=64)
    v1 = provider.embed_text("Calculate velocity of a car")
    v2 = provider.embed_text("Calculate velocity of an automobile")
    v3 = provider.embed_text("Complete matrix multiplication using linear algebra")

    assert len(v1) == 64
    assert len(v2) == 64
    assert len(v3) == 64

    # Calculate cosine similarity
    import numpy as np
    sim_1_2 = float(np.dot(v1, v2))
    sim_1_3 = float(np.dot(v1, v3))

    # Similar text should have higher similarity score than dissimilar text
    assert sim_1_2 > sim_1_3


def test_mock_vector_store():
    store = MockVectorStore()
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.9, 0.1, 0.0, 0.0]
    v3 = [0.0, 0.0, 1.0, 0.0]

    store.add("item1", v1, {"text": "item 1"})
    store.add("item2", v2, {"text": "item 2"})
    store.add("item3", v3, {"text": "item 3"})

    results = store.search(v1, top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "item1"
    assert results[0]["similarity"] > 0.99
    assert results[1]["id"] == "item2"

    store.clear()
    assert len(store.search(v1, top_k=2)) == 0
