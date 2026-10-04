"""
Repository layer managing persistence for API routes.
Encapsulates database access behind clean service methods.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from packages.schemas.models import GenerationRequest, GenerationResponse, GenerationMetrics, ReviewItem
from apps.api.models import GenerationRunDB, QuestionVariationDB, ReviewItemDB


class GenerationRepository:
    """Repository handling CRUD operations for generation runs and variations."""

    def __init__(self, db: Session):
        self.db = db

    def save_run(
        self,
        request: GenerationRequest,
        response: GenerationResponse,
        metrics: GenerationMetrics,
        reviews: List[ReviewItem]
    ) -> GenerationRunDB:
        run_record = GenerationRunDB(
            seed_question=request.seed_question,
            domain=request.domain,
            requested_count=metrics.requested_count,
            accepted_count=metrics.accepted_count,
            duplicate_count=metrics.duplicate_count,
            duplicate_rate=metrics.duplicate_rate,
            generation_time_seconds=metrics.generation_time_seconds
        )
        self.db.add(run_record)
        self.db.flush()

        for v in response.variations:
            var_record = QuestionVariationDB(
                run_id=run_record.id,
                question=v.question,
                answer_key=v.answer_key,
                difficulty=v.difficulty,
                domain=request.domain
            )
            self.db.add(var_record)

        for rev in reviews:
            rev_record = ReviewItemDB(
                question_text=rev.variation.question,
                answer_key=rev.variation.answer_key,
                difficulty=rev.variation.difficulty,
                reason=rev.reason,
                confidence_score=rev.confidence_score,
                status=rev.status
            )
            self.db.add(rev_record)

        self.db.commit()
        self.db.refresh(run_record)
        return run_record

    def get_run(self, run_id: str) -> Optional[GenerationRunDB]:
        return self.db.query(GenerationRunDB).filter(GenerationRunDB.id == run_id).first()

    def list_pending_reviews() -> List[ReviewItemDB]:
        return self.db.query(ReviewItemDB).filter(ReviewItemDB.status == "pending").all()
