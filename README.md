# AMIGO — Assignment Question Iteration & Variation Generation System

AMIGO is a production-ready, modular backend system developed for the **AMYPO National Hackathon 2026 Problem Statement 8 (PS8)**: *"Assignment Question Iteration & Variation Generation System"*.

The core objective is to generate multiple distinct, pedagogically equivalent question variations from a seed question while preserving learning objectives, difficulty levels, and generating accurate answer keys.

---

## 🏛️ Architecture Overview

The system is structured as a decoupled multi-app workspace with shared packages and clean dependency injection:

```
amigo/
├── apps/
│   ├── api/          # FastAPI HTTP service, SQLite database persistence, repository pattern
│   ├── generator/    # Seed question parsing, objective extraction, variation & answer generation
│   ├── evaluator/    # Difficulty scoring, equivalence validation, duplicate detection, answer key validation
│   └── orchestrator/ # Workflow pipeline coordinator binding generation, evaluation & review queue
├── packages/
│   ├── schemas/      # Shared Pydantic data contracts (SeedQuestion, GenerationRequest, etc.)
│   └── common/       # Provider abstractions (LLMProvider, EmbeddingProvider, VectorStore) & Mock Providers
├── tests/            # Automated pytest test suite covering all modules & API contracts
├── docs/             # Technical design & architecture documentation
├── models/           # Local open-weight model directory
├── data/             # SQLite database and vector storage directory
└── pyproject.toml    # Packaging & dependency configuration
```

---

## 🧩 Module Responsibilities & Decoupling Rules

- **`apps/api`**: Owns persistence and HTTP contract endpoints. Contains SQLite database models (`GenerationRunDB`, `QuestionVariationDB`, `ReviewItemDB`), FastAPI routes, and repository abstractions (`GenerationRepository`).
- **`apps/generator`**: Pure computational module responsible for parsing seed questions (`QuestionParser`), extracting pedagogical learning objectives (`LearningObjectiveAnalyzer`), generating candidate variations (`VariationGenerator`), and building solution answer keys (`AnswerKeyGenerator`). Does **not** contain database code.
- **`apps/evaluator`**: Independently testable evaluation engine providing difficulty analysis (`DifficultyAnalyzer`), difficulty equivalence validation (`DifficultyEquivalenceValidator`), near-duplicate detection via vector similarity (`DuplicateDetector`), and answer key validation (`AnswerValidator`).
- **`apps/orchestrator`**: Workflow pipeline manager (`GenerationOrchestrator`) coordinating seed question parsing -> objective extraction -> variation generation -> quality evaluation -> duplicate filtering -> review queue routing -> telemetry metrics computation. Contains zero model- or provider-specific code.
- **`packages/common/providers`**: Provider interfaces (`LLMProvider`, `EmbeddingProvider`, `VectorStore`) preventing architectural lock-in to Ollama, OpenAI, Gemini, or any single model family. Includes a deterministic `MockLLMProvider`, `MockEmbeddingProvider`, and `MockVectorStore` for offline development and automated testing.

---

## 🚀 How to Run the Backend

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.6)

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Start the FastAPI API Server
```bash
python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```
The server will run at `http://localhost:8000`.

Interactive Swagger API Documentation is available at:
`http://localhost:8000/docs`

---

## 🧪 How to Run Tests

Execute the automated test suite with pytest:

```bash
pytest -v
```

The test suite validates:
- Pydantic schema creation & constraints (`tests/test_schemas.py`)
- Provider interfaces & mock determinism (`tests/test_providers.py`)
- Question parser, objective analyzer, variation generator, and answer generator (`tests/test_generator.py`)
- Difficulty analyzer, equivalence validator, duplicate detector, answer validator, and variation evaluator (`tests/test_evaluator.py`)
- Orchestrator pipeline workflow generating 60 variations (`tests/test_orchestrator.py`)
- Official PS8 API contract endpoints: `POST /api/v1/generate`, `GET /api/v1/domains`, `GET /api/v1/health` (`tests/test_api.py`)

---

## 📋 PS8 API Endpoints Contract

### 1. Generate Variations
- **Endpoint**: `POST /api/v1/generate`
- **Request Body**:
```json
{
  "seed_question": "Calculate the velocity of a vehicle moving 100m in 5 seconds.",
  "domain": "Physics",
  "count": 60
}
```
- **Response Body**:
```json
{
  "variations": [
    {
      "question": "[Variation #1] For mass sliding on an inclined plane with friction coefficient = 11.70 and secondary factor = 6.15, Calculate the resulting stopping distance.",
      "answer_key": "Solution: 12.50\nExplanation: Step 1: Identify given parameters. Step 2: Apply core relationship for Physics. Result = 12.50.",
      "difficulty": 0.4
    }
  ],
  "duplicate_rate": 0.0
}
```

### 2. List Selectable Subject Domains
- **Endpoint**: `GET /api/v1/domains`
- **Response**:
```json
{
  "domains": [
    { "id": "cs", "name": "Computer Science", "description": "Algorithms, data structures, software architecture, systems" },
    { "id": "physics", "name": "Physics", "description": "Classical mechanics, electromagnetism, thermodynamics, quantum mechanics" },
    { "id": "math", "name": "Mathematics", "description": "Calculus, linear algebra, probability, differential equations" },
    { "id": "ee", "name": "Electrical Engineering", "description": "Circuit analysis, signal processing, digital logic, power systems" },
    { "id": "chem", "name": "Chemistry", "description": "Physical chemistry, stoichiometry, thermodynamics, reaction kinetics" }
  ]
}
```

### 3. Service Health Check
- **Endpoint**: `GET /api/v1/health`
- **Response**:
```json
{
  "status": "healthy",
  "phase": "Phase 1 - Backend Foundation & Shared Contracts",
  "version": "1.0.0"
}
```

---

## ⚠️ Current Limitations (Phase 1)

1. **Generation Provider**: Uses the deterministic offline `MockLLMProvider` and character n-gram `MockEmbeddingProvider` for Phase 1 testing and development.
2. **Local Open-Weight Model Integration**: Ollama / HuggingFace local open-weight model integration (e.g. Qwen / Llama 3 / Mistral) will be plugged into the `LLMProvider` interface in Phase 2.
3. **Domain Knowledge RAG**: Domain concept knowledge bases and vector store persistence across sessions will be integrated in Phase 2 via RAG pipelines.
4. **Frontend UI**: Frontend (web interface) is deferred to later phases as mandated by project scope rules.

---

## 🔮 Phase 2 Roadmap

1. Implement local open-weight model provider (`LocalOllamaProvider`, `HuggingFaceLocalProvider`) adhering to the PS8 no-paid-API requirement.
2. Implement vector store persistence (FAISS/Chroma) and RAG retriever for domain formula verification.
3. Add CSV / JSON bulk export modules.
4. Implement instructor review UI queue for low-confidence variations.
