"""
Generation Orchestrator implementation.
Coordinates seed question parsing, objective extraction, constraint representation,
variation generation, quality evaluation, duplicate filtering, and review queue routing.
Includes structured telemetry logging.
"""

import logging
import time
from typing import List, Tuple, Optional
from uuid import uuid4

from packages.schemas.models import (
    SeedQuestion,
    GenerationRequest,
    GenerationResponse,
    VariationOutput,
    QuestionVariation,
    GenerationMetrics,
    ReviewItem,
)
from packages.common.providers.interfaces import EmbeddingProvider, VectorStore
from packages.common.config import settings
from apps.generator.interfaces import QuestionParser, LearningObjectiveAnalyzer, VariationGenerator
from apps.evaluator.interfaces import VariationEvaluator, DuplicateDetector
from apps.orchestrator.interfaces import OrchestratorInterface

logger = logging.getLogger("amigo.orchestrator")


class GenerationOrchestrator(OrchestratorInterface):
    """Pipeline orchestrator binding generator, evaluator, vector store, and telemetry logging."""

    def __init__(
        self,
        parser: QuestionParser,
        objective_analyzer: LearningObjectiveAnalyzer,
        generator: VariationGenerator,
        evaluator: VariationEvaluator,
        duplicate_detector: DuplicateDetector,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        max_regeneration_attempts: int = 5
    ):
        self.parser = parser
        self.objective_analyzer = objective_analyzer
        self.generator = generator
        self.evaluator = evaluator
        self.duplicate_detector = duplicate_detector
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.max_regeneration_attempts = max_regeneration_attempts

    def orchestrate_generation(
        self, request: GenerationRequest
    ) -> Tuple[GenerationResponse, GenerationMetrics, List[ReviewItem]]:
        start_time = time.time()
        provider_name = settings.llm_provider
        model_name = settings.local_model if provider_name == "local" else "MockLLM"

        logger.info(
            f"Starting generation run: provider={provider_name}, model={model_name}, "
            f"domain={request.domain}, requested_count={request.count}"
        )

        # Clear vector store for clean generation run
        self.vector_store.clear()

        # Step 1: Create Seed Question domain object
        seed = SeedQuestion(
            text=request.seed_question,
            domain=request.domain,
            difficulty=request.target_difficulty,
            metadata={"id": str(uuid4())}
        )

        # Index seed question vector in vector store
        seed_vec = self.embedding_provider.embed_text(seed.text)
        self.vector_store.add(seed.metadata["id"], seed_vec, {"text": seed.text, "type": "seed"})

        # Step 2: Parse Seed Question & Extract Constraints
        parsed = self.parser.parse(seed)

        # Step 3: Extract Learning Objective
        objective = self.objective_analyzer.extract_objective(parsed)

        # Step 4: Iterative Generation & Evaluation Loop (Batch Incremental Execution)
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

        while len(accepted_variations) < target_count and current_attempt < self.max_regeneration_attempts:
            needed = target_count - len(accepted_variations)
            
            raw_candidates = self.generator.generate_variations(
                seed=seed,
                parsed=parsed,
                objective=objective,
                count=needed,
                target_domain=request.domain,
                start_index=total_generated
            )
            
            total_generated += len(raw_candidates)
            current_attempt += 1

            for candidate in raw_candidates:
                val_result = self.evaluator.evaluate_variation(
                    variation=candidate,
                    seed=seed,
                    existing_variations=accepted_variations
                )
                candidate.validation_result = val_result

                if not val_result.objective_valid:
                    objective_failures += 1
                if not val_result.answer_valid:
                    answer_failures += 1

                cand_vec = self.embedding_provider.embed_text(candidate.question)
                dup_res = self.duplicate_detector.check_duplicate(
                    candidate_text=candidate.question,
                    existing_texts=[v.question for v in accepted_variations],
                    threshold=settings.duplicate_threshold
                )
                candidate.duplicate_result = dup_res

                if dup_res.is_duplicate:
                    total_duplicates += 1

                if val_result.is_valid:
                    self.vector_store.add(candidate.id, cand_vec, {"text": candidate.question, "type": "variation"})
                    accepted_variations.append(candidate)
                else:
                    total_rejected += 1
                    if not dup_res.is_duplicate and val_result.answer_valid and not val_result.is_difficulty_equivalent:
                        total_low_confidence += 1
                        rev_item = ReviewItem(
                            id=str(uuid4()),
                            variation=candidate,
                            reason="Difficulty shift flag",
                            confidence_score=0.75,
                            status="pending"
                        )
                        review_items.append(rev_item)

                if len(accepted_variations) >= target_count:
                    break

        elapsed_time = round(time.time() - start_time, 3)
        dup_rate = round(total_duplicates / max(1, total_generated), 4)

        logger.info(
            f"Generation run completed in {elapsed_time}s: generated={total_generated}, "
            f"accepted={len(accepted_variations)}, rejected={total_rejected}, "
            f"duplicates={total_duplicates}, objective_failures={objective_failures}, "
            f"answer_failures={answer_failures}"
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
