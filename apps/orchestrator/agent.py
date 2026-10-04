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
            f"[MASTER] Generation request received: provider={provider_name}, model={model_name}, "
            f"domain={request.domain}, count={request.count}"
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
        logger.info(f"[SEED] Seed question analyzed: topic={analysis.topic}, bloom_level={analysis.bloom_level}")

        # Step 2: Retrieve RAG Knowledge Context
        rag_context: RAGKnowledgeContext = self.knowledge_provider.get_context(
            domain=request.domain,
            topic=analysis.topic,
            bloom_level=analysis.bloom_level
        )
        logger.info(f"[RAG] Knowledge context prepared for domain={request.domain}, topic={analysis.topic}")

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
            current_start_index = len(accepted_variations)

            # Generate blueprints via VariationPlanner
            blueprints = self.planner.plan_variations(
                analysis=analysis,
                count=needed,
                rag_context=rag_context,
                start_index=current_start_index
            )
            logger.info(f"[PLANNER] {len(blueprints)} authoritative blueprints created")

            # Parse seed and extract objective for variation generator
            parsed_seed = self.seed_analyzer.parser.parse(seed) if hasattr(self.seed_analyzer, "parser") else None
            extracted_obj = self.seed_analyzer.objective_analyzer.extract_objective(parsed_seed) if (parsed_seed and hasattr(self.seed_analyzer, "objective_analyzer")) else None

            # Generate candidate variations passing authoritative blueprints
            raw_candidates = self.generator.generate_variations(
                seed=seed,
                parsed=parsed_seed,
                objective=extracted_obj,
                count=needed,
                target_domain=request.domain,
                start_index=current_start_index,
                blueprints=blueprints
            )
            logger.info(f"[GENERATOR] {len(raw_candidates)} candidate question wording variations generated")
            logger.info(f"[ANSWER] Deterministic answer keys generated for {len(raw_candidates)} candidate variations")

            total_generated += len(raw_candidates)
            current_attempt += 1

            for offset, candidate in enumerate(raw_candidates):
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
                    logger.info(f"[QUALITY] Candidate variation {len(accepted_variations)} accepted")
                else:
                    total_rejected += 1
                    logger.warning(f"[EVALUATOR] Candidate rejected: reasons={val_result.reasons}")
                    # Targeted Regeneration: Attempt blueprint repair based on failure reasons
                    if hasattr(self.planner, "repair_blueprint") and offset < len(blueprints):
                        try:
                            failed_bp = blueprints[offset]
                            repaired_bp = self.planner.repair_blueprint(failed_bp, val_result.reasons)
                            logger.info(f"[PLANNER] Blueprint repaired for rejected candidate due to {val_result.reasons}")
                            repaired_candidates = self.generator.generate_variations(
                                seed=seed,
                                parsed=parsed_seed,
                                objective=extracted_obj,
                                count=1,
                                target_domain=request.domain,
                                start_index=len(accepted_variations),
                                blueprints=[repaired_bp]
                            )
                            if repaired_candidates:
                                rep_cand = repaired_candidates[0]
                                rep_val_res = self.quality_gate.evaluate_candidate(
                                    candidate=rep_cand,
                                    seed=seed,
                                    existing_variations=accepted_variations,
                                    seed_bloom_level=analysis.bloom_level
                                )
                                rep_cand.validation_result = rep_val_res
                                if rep_val_res.is_valid:
                                    cand_vec = self.embedding_provider.embed_text(rep_cand.question)
                                    self.vector_store.add(rep_cand.id, cand_vec, {"text": rep_cand.question, "type": "variation"})
                                    accepted_variations.append(rep_cand)
                                    logger.info(f"[QUALITY] Repaired candidate variation {len(accepted_variations)} accepted")
                                    continue
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
            f"[MASTER] Generation run complete: generated={total_generated}, accepted={len(accepted_variations)}, "
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

        # Compute strategy distribution across accepted variations
        strategy_dist: Dict[str, int] = {}
        for v in accepted_variations[:target_count]:
            try:
                bp_dict = json.loads(v.context_changes) if v.context_changes else {}
                strat = bp_dict.get("strategy_name", "direct_calculation")
            except Exception:
                strat = "direct_calculation"
            strategy_dist[strat] = strategy_dist.get(strat, 0) + 1

        obj_preservation_rate = round(1.0 - (objective_failures / max(1, total_generated)), 4)
        ans_correctness_rate = round(1.0 - (answer_failures / max(1, total_generated)), 4)
        diff_equiv_rate = round(1.0 - (total_low_confidence / max(1, total_generated)), 4)

        llm_provider = getattr(self.generator, "llm_provider", None)
        if hasattr(llm_provider, "invocation_count") and llm_provider.invocation_count > 0:
            llm_invocations = llm_provider.invocation_count
            avg_llm_latency = round(llm_provider.average_latency, 4)
            total_llm_latency = round(llm_provider.total_latency, 4)
        else:
            llm_invocations = total_generated
            avg_llm_latency = round(elapsed_time / max(1, llm_invocations), 4)
            total_llm_latency = elapsed_time

        logger.info(
            f"Master Agent Telemetry Summary: provider={provider_name}, model={model_name}, "
            f"requested={target_count}, accepted={len(response_variations)}, rejected={total_rejected}, "
            f"regeneration_count={max(0, current_attempt - 1)}, wall_clock_time={elapsed_time:.4f}s, "
            f"local_llm_invocation_count={llm_invocations}, average_llm_latency={avg_llm_latency:.4f}s, "
            f"total_llm_latency={total_llm_latency:.4f}s"
        )

        metrics = GenerationMetrics(
            requested_count=target_count,
            generated_count=total_generated,
            accepted_count=len(response_variations),
            rejected_count=total_rejected,
            regeneration_count=max(0, current_attempt - 1),
            duplicate_count=total_duplicates,
            duplicate_rate=dup_rate,
            low_confidence_count=total_low_confidence,
            low_confidence_review_count=len(review_items),
            objective_failure_count=objective_failures,
            answer_failure_count=answer_failures,
            generation_time_seconds=elapsed_time,
            total_wall_clock_time=elapsed_time,
            llm_invocation_count=llm_invocations,
            average_llm_latency=avg_llm_latency,
            objective_preservation_rate=obj_preservation_rate,
            answer_key_correctness_rate=ans_correctness_rate,
            difficulty_equivalence_rate=diff_equiv_rate,
            variation_strategy_distribution=strategy_dist
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
