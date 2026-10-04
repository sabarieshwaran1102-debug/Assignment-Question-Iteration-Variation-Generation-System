"""
Golden Dataset Evaluation Command.
Executes AMIGO orchestrator pipeline against the PS8 golden evaluation dataset and reports:
- Objective Preservation Rate
- Answer-Key Validity Rate
- Duplicate Rate
- Average Difficulty Difference
- Valid Variation Rate
- Generation Execution Time
"""

import json
import os
import sys
import time
from typing import Dict, List, Any

from packages.schemas.models import GenerationRequest
from packages.common.config import settings


def evaluate_golden_dataset(count_per_seed: int = 10) -> Dict[str, Any]:
    from apps.api.dependencies import get_orchestrator

    golden_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "golden", "seed_questions.json"
    )

    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        seed_items = json.load(f)

    orchestrator = get_orchestrator()
    provider_name = settings.llm_provider
    model_name = settings.local_model if provider_name == "local" else "MockLLM"

    print("=" * 80)
    print("AMIGO PS8 GOLDEN DATASET EVALUATION BENCHMARK")
    print(f"Provider: {provider_name.upper()} | Model: {model_name} | Seeds: {len(seed_items)}")
    print("=" * 80)

    total_requested = 0
    total_generated = 0
    total_accepted = 0
    total_rejected = 0
    total_duplicates = 0
    total_objective_failures = 0
    total_answer_failures = 0
    diff_differences: List[float] = []

    start_total_time = time.time()
    per_seed_results = []

    for idx, item in enumerate(seed_items, 1):
        seed_text = item["seed_question"]
        domain = item["domain"]
        target_diff = item.get("difficulty_expectation", 0.5)

        print(f"\n[{idx}/{len(seed_items)}] Domain: {domain} | Seed: '{seed_text[:55]}...'")

        req = GenerationRequest(
            seed_question=seed_text,
            domain=domain,
            count=count_per_seed,
            target_difficulty=target_diff
        )

        t0 = time.time()
        response, metrics, review_items = orchestrator.orchestrate_generation(req)
        elapsed = time.time() - t0

        total_requested += metrics.requested_count
        total_generated += metrics.generated_count
        total_accepted += metrics.accepted_count
        total_rejected += metrics.rejected_count
        total_duplicates += metrics.duplicate_count
        total_objective_failures += metrics.objective_failure_count
        total_answer_failures += metrics.answer_failure_count

        for var in response.variations:
            diff_differences.append(abs(var.difficulty - target_diff))

        seed_valid_rate = round(metrics.accepted_count / max(1, metrics.generated_count), 4)

        print(f"  Generated: {metrics.generated_count} | Accepted: {metrics.accepted_count} | "
              f"Duplicates: {metrics.duplicate_count} | Time: {elapsed:.2f}s | Valid Rate: {seed_valid_rate * 100:.1f}%")

        per_seed_results.append({
            "id": item.get("id"),
            "domain": domain,
            "seed_question": seed_text,
            "metrics": metrics.model_dump(),
            "valid_variation_rate": seed_valid_rate
        })

    total_elapsed_time = round(time.time() - start_total_time, 2)
    
    valid_variation_rate = round(total_accepted / max(1, total_generated), 4)
    objective_preservation_rate = round(1.0 - (total_objective_failures / max(1, total_generated)), 4)
    answer_key_validity_rate = round(1.0 - (total_answer_failures / max(1, total_generated)), 4)
    duplicate_rate = round(total_duplicates / max(1, total_generated), 4)
    avg_difficulty_diff = round(sum(diff_differences) / max(1, len(diff_differences)), 4)

    summary_report = {
        "provider": provider_name,
        "model_name": model_name,
        "total_seeds": len(seed_items),
        "total_requested": total_requested,
        "total_generated": total_generated,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_duplicates": total_duplicates,
        "total_objective_failures": total_objective_failures,
        "total_answer_failures": total_answer_failures,
        "objective_preservation_rate": objective_preservation_rate,
        "answer_key_validity_rate": answer_key_validity_rate,
        "duplicate_rate": duplicate_rate,
        "average_difficulty_difference": avg_difficulty_diff,
        "valid_variation_rate": valid_variation_rate,
        "total_generation_time_seconds": total_elapsed_time,
        "seed_details": per_seed_results
    }

    print("\n" + "=" * 80)
    print("GOLDEN BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"Objective Preservation Rate : {objective_preservation_rate * 100:.2f}%")
    print(f"Answer-Key Validity Rate   : {answer_key_validity_rate * 100:.2f}%")
    print(f"Duplicate Rate              : {duplicate_rate * 100:.2f}% (Target: < 10%)")
    print(f"Average Difficulty Diff     : ±{avg_difficulty_diff:.4f}")
    print(f"Valid Variation Rate        : {valid_variation_rate * 100:.2f}%")
    print(f"Total Execution Time        : {total_elapsed_time:.2f} seconds")
    print("=" * 80)

    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "golden_evaluation_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    return summary_report


if __name__ == "__main__":
    evaluate_golden_dataset()
