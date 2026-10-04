# AMIGO Data Directory

This directory manages database persistence, evaluation datasets, fine-tuning trajectory logs, and export artifacts for the **AMIGO** platform.

---

## 📁 Directory Structure & File Contents

```
data/
├── golden/
│   └── seed_questions.json        # Benchmark golden seed questions for evaluation
├── phase2_results/
│   ├── physics_15_variations.csv # Generated 15-variation physics dataset in CSV format
│   └── physics_15_variations.json# Generated 15-variation physics dataset in JSON format
├── amigo.db                       # SQLite relational database storing runs, variations, and reviews
├── fine_tuning_dataset.jsonl      # Trajectory dataset logging generation runs for future fine-tuning
└── golden_evaluation_report.json   # Automated evaluation quality report over golden seed questions
```

---

## 🗄️ Database Schema (`amigo.db`)

Managed by SQLAlchemy (`apps/api/database.py`, `apps/api/models.py`) and repository patterns (`apps/api/repository.py`):

1. **`generation_runs` (`GenerationRunDB`)**:
   - Stores run metadata, request parameters, timing metrics (`wall_clock_time`, `llm_latency`), duplicate rates, and strategy distributions.
2. **`question_variations` (`QuestionVariationDB`)**:
   - Stores individual accepted question variations, calculated answer keys, assigned difficulty scores, context changes, and validation flags.
3. **`review_items` (`ReviewItemDB`)**:
   - Stores low-confidence variation items flagged for instructor review (e.g. non-duplicate difficulty shifts).

---

## 📝 Fine-Tuning Dataset Trajectory (`fine_tuning_dataset.jsonl`)

The `MasterAgent` automatically appends candidate generation trajectories to `data/fine_tuning_dataset.jsonl` during runtime.

Each line contains a JSON record:
```json
{
  "seed_question": "A car travels a distance of 150 meters in 5 seconds. What is the velocity of the car?",
  "blueprint": { "context": "..." },
  "generated_question": "A cyclist travels 240 meters in 12 seconds. Calculate the velocity of the cyclist.",
  "accepted": true,
  "rejection_reasons": [],
  "evaluator_scores": { "difficulty": 0.5 },
  "answer_correctness": true,
  "duplicate_score": 0.0,
  "difficulty_score": 0.5,
  "bloom_level": "Analyze",
  "timestamp": 1759560000.0
}
```

---

## 🏆 Golden Dataset & Benchmarks (`golden/`)

- **`golden/seed_questions.json`**: Standardized set of seed questions across academic domains (Physics, Computer Science, Mathematics, Electrical Engineering, Chemistry).
- **`golden_evaluation_report.json`**: Evaluation report validating learning objective preservation, difficulty equivalence, and duplicate rates across golden seeds.
