# AMIGO Phase 1 Architecture Overview

## Overview
AMIGO is an automated Assignment Question Iteration & Variation Generation System designed for AMYPO National Hackathon 2026 Problem Statement 8 (PS8).

## Modular Structure

```
apps/
  api/          # HTTP REST API, persistence, request/response contracts
  generator/    # Seed question parsing, objective extraction, variation & answer generation
  evaluator/    # Difficulty analysis, equivalence checking, duplicate detection, answer validation
  orchestrator/ # Workflow coordination across parsing -> generation -> evaluation -> filtering

packages/
  schemas/      # Shared Pydantic data contracts (SeedQuestion, GenerationRequest, etc.)
  common/       # Abstract provider interfaces (LLM, Embedding, VectorStore) and mock implementations

tests/          # Comprehensive unit test suite covering all modules and API contracts
docs/           # Technical documentation
models/         # Local open-weight model directory
data/           # SQLite persistence directory
```

## Architectural Decoupling Rules

1. **API owns persistence**: The SQLite database and repositories belong strictly to `apps/api`.
2. **Generator does NOT own persistence**: `apps/generator` is a pure computation module.
3. **Evaluator is independently testable**: All evaluator components can be unit-tested without external APIs or databases.
4. **Orchestrator coordinates**: `apps/orchestrator` coordinates the workflow and contains zero provider-specific code.
5. **Provider interfaces**: Provider abstractions (`LLMProvider`, `EmbeddingProvider`, `VectorStore`) prevent lock-in to Ollama, OpenAI, Gemini, or any specific framework.
