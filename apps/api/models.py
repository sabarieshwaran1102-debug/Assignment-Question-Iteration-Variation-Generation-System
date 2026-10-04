"""
SQLAlchemy ORM models for persistent storage of generation runs, variations, and review items.
"""

from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship

from apps.api.database import Base


class GenerationRunDB(Base):
    """Database model for a generation run session."""
    __tablename__ = "generation_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    seed_question = Column(Text, nullable=False)
    domain = Column(String(100), nullable=False)
    requested_count = Column(Integer, nullable=False)
    accepted_count = Column(Integer, nullable=False)
    duplicate_count = Column(Integer, nullable=False)
    duplicate_rate = Column(Float, nullable=False)
    generation_time_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    variations = relationship("QuestionVariationDB", back_populates="run", cascade="all, delete-orphan")


class QuestionVariationDB(Base):
    """Database model for an individual question variation."""
    __tablename__ = "question_variations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id = Column(String(36), ForeignKey("generation_runs.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer_key = Column(Text, nullable=False)
    difficulty = Column(Float, nullable=False)
    domain = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    run = relationship("GenerationRunDB", back_populates="variations")


class ReviewItemDB(Base):
    """Database model for low-confidence review queue items."""
    __tablename__ = "review_items"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    question_text = Column(Text, nullable=False)
    answer_key = Column(Text, nullable=False)
    difficulty = Column(Float, nullable=False)
    reason = Column(String(255), nullable=False)
    confidence_score = Column(Float, nullable=False)
    status = Column(String(50), default="pending")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
