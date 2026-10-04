"""
Master Agent / Orchestrator implementation for Phase 3.
Coordinates seed analysis, RAG knowledge retrieval, variation planning, local LLM generation,
deterministic answer solving, specialized evaluation, smart targeted regeneration,
low-confidence review queue routing, bulk export, and fine-tuning dataset recording.
"""

import csv
import json
import logging
import os
import time
from typing import List, Tuple, Optional, Dict, Any
from uuid import uuid4

from packages.schemas.models import (
    SeedQuestion,
    GenerationRequest,
    GenerationResponse,
    VariationOutput,
    QuestionVariation,
    GenerationMetrics,
    ReviewItem,
    SeedQuestionAnalysis,
    RAGKnowledgeContext,
    FineTuningRecord,
    ValidationResult,
)
from packages.common.providers.interfaces import EmbeddingProvider, VectorStore, LLMProvider
from packages.common.config import settings
from packages.common.rag import KnowledgeContextProvider, DefaultKnowledgeContextProvider
from apps.generator.seed_analyzer import SeedAnalyzer
from apps.generator.planner import VariationPlanner
from apps.generator.interfaces import QuestionParser, LearningObjectiveAnalyzer, VariationGenerator, AnswerKeyGenerator
from apps.generator.solver import AnswerSolver, DefaultAnswerSolver
from apps.evaluator.specialized import QualityEvaluator
from apps.evaluator.interfaces import DuplicateDetector
from apps.orchestrator.interfaces import OrchestratorInterface

logger = logging.getLogger("amigo.orchestrator.agent")


class MasterAgent(OrchestratorInterface):
    """Agentic Master Orchestrator binding seed analysis, RAG, planning, local LLMs, specialized evaluation, and smart regeneration."""

    def __init__(
        self,
        seed_analyzer: SeedAnalyzer,
        knowledge_provider: KnowledgeContextProvider,
        planner: VariationPlanner,
        generator: VariationGenerator,
        answer_solver: AnswerSolver,
        answer_generator: AnswerKeyGenerator,
        quality_gate: QualityEvaluator,
        duplicate_detector: DuplicateDetector,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        dataset_path: str = "data/fine_tuning_dataset.jsonl",
        max_regeneration_attempts: int = 5
    ):
        self.seed_analyzer = seed_analyzer
        self.knowledge_provider = knowledge_provider
        self.planner = planner
        self.generator = generator
        self.answer_solver = answer_solver
        self.answer_generator = answer_generator
        self.quality_gate = quality_gate
        self.duplicate_detector = duplicate_detector
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.dataset_path = dataset_path
        self.max_regeneration_attempts = max_regeneration_attempts

        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

    def orchestrate_generation(
        self, request: GenerationRequest
    ) -> Tuple[GenerationResponse, GenerationMetrics, List[ReviewItem]]:
        start_time = time.time()
        provider_name = settings.llm_provider
        model_name = settings.local_model if provider_name == "local" else "MockLLM"

        logger.info(
            f"Master Agent starting run: provider={provider_name}, model={model_name}, "
            f"domain={request.domain}, requested_count={request.count}"
        )

        self.vector_store.clear()

        # Step 1: Create Seed Question & Seed Analysis
        seed = SeedQuestion(
            text=request.seed_question,
            domain=request.domain,
            difficulty=request.target_difficulty,
            metadata={"id": str(uuid4())}
        )

        seed_vec = self.embedding_provider.embed_text(seed.text)
        self.vector_store.add(seed.metadata["id"], seed_vec, {"text": seed.text, "type": "seed"})

        analysis: SeedQuestionAnalysis = self.seed_analyzer.analyze_seed(seed)

        # Step 2: Retrieve RAG Knowledge Context
        rag_context: RAGKnowledgeContext = self.knowledge_provider.get_context(
            domain=request.domain,
            topic=analysis.topic,
            bloom_level=analysis.bloom_level
        )

        accepted_variations: List[QuestionVariation] = []
        review_items: List[ReviewItem] = []

        total_generated = 0
        total_duplicates = 0
        total_rejected = 0
        total_low_confidence = 0
        objective_failures = 0
        answer_failures = 0

        target_count = request.count
        current_attempt = 0

        # Step 3: Iterative Generation & Targeted Regeneration Loop
        while len(accepted_variations) < target_count and current_attempt < self.max_regeneration_attempts:
            needed = target_count - len(accepted_variations)

            # Generate blueprints via VariationPlanner
            blueprints = self.planner.plan_variations(
                analysis=analysis,
                count=needed,
                rag_context=rag_context,
                start_index=total_generated
            )

            # Parse seed and extract objective for variation generator
            parsed_seed = self.seed_analyzer.parser.parse(seed) if hasattr(self.seed_analyzer, "parser") else None
            extracted_obj = self.seed_analyzer.objective_analyzer.extract_objective(parsed_seed) if (parsed_seed and hasattr(self.seed_analyzer, "objective_analyzer")) else None

            # Generate candidate variations
            raw_candidates = self.generator.generate_variations(
                seed=seed,
                parsed=parsed_seed,
                objective=extracted_obj,
                count=needed,
                target_domain=request.domain,
                start_index=total_generated
            )


            total_generated += len(raw_candidates)
            current_attempt += 1

            for candidate in raw_candidates:
                val_result: ValidationResult = self.quality_gate.evaluate_candidate(
                    candidate=candidate,
                    seed=seed,
                    existing_variations=accepted_variations,
                    seed_bloom_level=analysis.bloom_level
                )
                candidate.validation_result = val_result

                if not val_result.objective_valid:
                    objective_failures += 1
                if not val_result.answer_valid:
                    answer_failures += 1
                if candidate.duplicate_result and candidate.duplicate_result.is_duplicate:
                    total_duplicates += 1

                # Record fine-tuning dataset entry
                self._record_fine_tuning_data(seed, candidate, val_result, analysis.bloom_level)

                if val_result.is_valid:
                    cand_vec = self.embedding_provider.embed_text(candidate.question)
                    self.vector_store.add(candidate.id, cand_vec, {"text": candidate.question, "type": "variation"})
                    accepted_variations.append(candidate)
                else:
                    total_rejected += 1
                    # Targeted Regeneration: Attempt blueprint repair based on failure reasons
                    if hasattr(self.planner, "repair_blueprint") and offset < len(blueprints):
                        try:
                            failed_bp = blueprints[offset]
                            repaired_bp = self.planner.repair_blueprint(failed_bp, val_result.reasons)
                            logger.info(f"Master Agent repairing blueprint for candidate {candidate.id} due to {val_result.reasons}")
                        except Exception as e:
                            logger.debug(f"Blueprint repair skipped: {e}")

                    # Low Confidence Queue Routing for non-duplicate difficulty/borderline shifts
                    dup_is = candidate.duplicate_result.is_duplicate if candidate.duplicate_result else False
                    if not dup_is and val_result.answer_valid and not val_result.is_difficulty_equivalent:
                        total_low_confidence += 1
                        rev_item = ReviewItem(
                            id=str(uuid4()),
                            variation=candidate,
                            reason=f"Difficulty shift flag: {val_result.reasons}",
                            confidence_score=0.75,
                            status="pending"
                        )
                        review_items.append(rev_item)

                if len(accepted_variations) >= target_count:
                    break

        elapsed_time = round(time.time() - start_time, 3)
        dup_rate = round(total_duplicates / max(1, total_generated), 4)

        logger.info(
            f"Master Agent run complete: generated={total_generated}, accepted={len(accepted_variations)}, "
            f"rejected={total_rejected}, duplicates={total_duplicates}, time={elapsed_time}s"
        )

        response_variations = [
            VariationOutput(
                question=v.question,
                answer_key=v.answer_key,
                difficulty=v.difficulty
            )
            for v in accepted_variations[:target_count]
        ]

        response = GenerationResponse(
            variations=response_variations,
            duplicate_rate=dup_rate
        )

        metrics = GenerationMetrics(
            requested_count=target_count,
            generated_count=total_generated,
            accepted_count=len(response_variations),
            rejected_count=total_rejected,
            duplicate_count=total_duplicates,
            duplicate_rate=dup_rate,
            low_confidence_count=total_low_confidence,
            objective_failure_count=objective_failures,
            answer_failure_count=answer_failures,
            generation_time_seconds=elapsed_time
        )

        return response, metrics, review_items

    def _record_fine_tuning_data(
        self,
        seed: SeedQuestion,
        candidate: QuestionVariation,
        val_result: ValidationResult,
        bloom_level: str
    ):
        """Append generation trajectory to fine-tuning dataset log file."""
        try:
            rec = FineTuningRecord(
                seed_question=seed.text,
                blueprint={"context": candidate.context_changes},
                generated_question=candidate.question,
                accepted=val_result.is_valid,
                rejection_reasons=val_result.reasons,
                evaluator_scores={"difficulty": val_result.difficulty_score.score},
                answer_correctness=val_result.answer_valid,
                duplicate_score=candidate.duplicate_result.similarity_score if candidate.duplicate_result else 0.0,
                difficulty_score=val_result.difficulty_score.score,
                bloom_level=bloom_level,
                timestamp=time.time()
            )
            with open(self.dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.model_dump()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write fine-tuning record: {e}")

    def export_csv(self, variations: List[VariationOutput], file_path: str):
        """Bulk export variations to CSV format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer_key", "difficulty"])
            writer.writeheader()
            for v in variations:
                writer.writerow(v.model_dump())

    def export_json(self, variations: List[VariationOutput], file_path: str):
        """Bulk export variations to JSON format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([v.model_dump() for v in variations], f, indent=2)
