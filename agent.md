# AMIGO — Agent Development Rules

## 1. Project Scope

AMIGO is based on AMYPO National Hackathon 2026 Problem Statement 8:

"Assignment Question Iteration & Variation Generation System"

The primary goal is to generate multiple distinct question variations from a seed question while preserving the learning objective and equivalent difficulty.

PS8 is the core scope of this project.

Do not make PS2 Hallucination Detection a core dependency of the PS8 generation pipeline.

---

## 2. Official PS8 Requirements

The system must:

- Accept a seed question.
- Accept a selectable domain.
- Generate up to 60 valid variations.
- Support at least 5 selectable domains.
- Preserve the learning objective.
- Preserve equivalent difficulty unless a difficulty shift is explicitly requested.
- Change surface context/scenario/use case.
- Change underlying numbers or methodology where applicable.
- Generate a correct answer key for every variation.
- Detect duplicate and near-duplicate questions.
- Keep near-duplicate rate below 10%.
- Provide low-confidence review handling.
- Support CSV and JSON export.
- Target generation of 60 variations in under 5 minutes.
- Avoid paid third-party LLM APIs.
- Use local/open-weight generation for the final system.

Required API endpoints:

POST /api/v1/generate
GET  /api/v1/domains
GET  /api/v1/health

Required /generate request:

{
  "seed_question": "string",
  "domain": "string",
  "count": 60
}

Required /generate response:

{
  "variations": [
    {
      "question": "string",
      "answer_key": "string",
      "difficulty": 0.0
    }
  ],
  "duplicate_rate": 0.0
}

---

## 3. Mandatory Components

The architecture must contain:

1. Domain Selector & Question Parser
2. Variation Generation Engine
3. Difficulty & Equivalence Validator
4. Duplicate / Near-Duplicate Detector
5. Automated Answer Key Generator
6. Bulk Export Module
7. Low-Confidence Review Queue

---

## 4. Architecture Rules

Use a modular architecture.

Initial structure:

apps/
  web/
  api/
  generator/
  evaluator/
  orchestrator/

packages/
  schemas/
  common/

tests/
docs/
models/
data/

The modules must have clear responsibilities.

### Generator

Responsible for:
- question understanding
- learning objective extraction
- variation generation
- answer generation

The generator must NOT own database persistence.

### Evaluator

Responsible for:
- difficulty analysis
- difficulty equivalence
- duplicate detection
- answer validation
- variation quality evaluation

Evaluator components must be independently testable.

### Orchestrator

Responsible for workflow coordination.

It must not contain implementation details of individual generators or evaluators.

### API

Responsible for:
- HTTP endpoints
- request validation
- response formatting
- persistence
- application-level coordination

### Frontend

Must communicate through the API.

Do not put generation logic inside React.

---

## 5. Model Abstraction

Never hard-code the application to one LLM.

Create interfaces such as:

LLMProvider
EmbeddingProvider
VectorStore

Possible implementations may include:

- Mock provider for development/testing
- Local/open-weight provider
- Ollama/local runtime
- HuggingFace/local model
- Future fine-tuned model

The final implementation must respect the PS8 no-paid-API constraint.

---

## 6. Mock Provider

A mock provider is allowed for development and automated tests.

However, it must produce deterministic but meaningful test variations.

Do NOT create fake UI output that pretends the real generation engine works.

---

## 7. Quality Requirements

Every generated variation must have:

- question
- answer_key
- difficulty
- validation status
- evaluation information internally

The pipeline should be able to reject and regenerate poor candidates.

Track:

- requested count
- generated count
- accepted count
- rejected count
- duplicate count
- duplicate rate
- low-confidence count
- generation time

---

## 8. API Compatibility

Do not change the official PS8 API contract unless there is a strong reason.

Additional internal/application endpoints are allowed.

Maintain backward compatibility.

---

## 9. Testing

Every new feature must include tests.

At minimum test:

- question parsing
- objective extraction
- variation generation
- difficulty calculation
- duplicate detection
- answer-key generation
- API validation
- generation workflow

Do not remove existing tests to make the project pass.

---

## 10. Development Rules

Before modifying code:

1. Inspect the existing repository.
2. Understand existing implementation.
3. Reuse useful code.
4. Avoid unnecessary rewrites.
5. Keep changes within the requested module.
6. Do not modify unrelated files.

Never create:

- pseudocode presented as implementation
- TODO-only implementations
- fake API responses
- hard-coded generated question lists
- fake AI/model claims
- unnecessary dependencies

---

## 11. Documentation

Document:

- architecture
- API
- model configuration
- generation pipeline
- evaluation pipeline
- setup
- testing
- limitations

Never claim that a model is trained or fine-tuned unless actual training/fine-tuning has been implemented.

---

## 12. Priority

When requirements conflict, prioritize:

1. Official PS8 requirements
2. Correctness
3. Evaluation quality
4. Maintainable architecture
5. Performance
6. UI polish

Build a working PS8 system before adding optional features.