"""
Domain and API Pydantic schemas for the AMIGO platform.
Contains shared data models for seed questions, variations, requests,
responses, evaluation results, metrics, and seed constraints.
"""

from typing import Dict, List, Optional, Any
from uuid import uuid4
from pydantic import BaseModel, Field


class SeedQuestion(BaseModel):
    """Seed question provided by the instructor to generate variations from."""
    text: str = Field(..., description="The original seed question text")
    domain: str = Field(..., description="Academic or subject domain (e.g., Computer Science, Physics)")
    difficulty: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional baseline difficulty (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g. topic, tags, course_code)")


class SeedConstraints(BaseModel):
    """Pedagogical and structural constraints extracted from a seed question."""
    subject_domain: str = Field(..., description="Academic domain")
    topic: str = Field(..., description="Specific subtopic (e.g., Kinematics, Search Trees)")
    learning_objective: str = Field(..., description="Preserved pedagogical learning objective")
    task_type: str = Field(..., description="Task classification (e.g. quantitative calculation, algorithmic reasoning)")
    required_knowledge: List[str] = Field(default_factory=list, description="Core concepts and prerequisites")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Extracted numerical or logical parameters")
    formulas: List[str] = Field(default_factory=list, description="Mathematical relationships or equations")
    expected_answer_type: str = Field(..., description="Answer format (e.g. numeric with units, step-by-step code)")
    difficulty_characteristics: Dict[str, Any] = Field(default_factory=dict, description="Complexity and Bloom's level factors")


class VariationBlueprint(BaseModel):
    """Structured variation blueprint guiding natural language question synthesis."""
    domain: str = Field(..., description="Subject domain")
    topic: str = Field(..., description="Specific subtopic")
    learning_objective: str = Field(..., description="Preserved pedagogical learning objective")
    formula: str = Field(..., description="Hard mathematical relationship equation")
    task_type: str = Field(default="quantitative calculation", description="Task classification")
    scenario: str = Field(..., description="Target entity/scenario (e.g. cyclist, train, runner)")
    target_variable: str = Field(default="velocity", description="Variable to calculate")
    known_variables: Dict[str, float] = Field(default_factory=dict, description="Numeric parameter values")
    units: Dict[str, str] = Field(default_factory=dict, description="Units for parameters")
    strategy_name: str = Field(default="direct_calculation", description="Variation strategy pattern applied")
    target_bloom_level: str = Field(default="Apply", description="Target Bloom's taxonomy level")


class SeedQuestionAnalysis(BaseModel):
    """Deep structural and cognitive analysis of a seed question."""
    domain: str = Field(..., description="Subject domain")
    topic: str = Field(..., description="Topic classification")
    subtopic: str = Field(..., description="Specific subtopic")
    learning_objective: str = Field(..., description="Preserved learning objective")
    bloom_level: str = Field(..., description="Bloom's Taxonomy classification (Remember..Create)")
    difficulty: float = Field(..., ge=0.0, le=1.0, description="Normalized baseline difficulty")
    question_type: str = Field(..., description="Classification (e.g. quantitative, conceptual)")
    concepts: List[str] = Field(default_factory=list, description="Extracted core concepts")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Extracted numerical or logical parameters")
    constraints: List[str] = Field(default_factory=list, description="Extracted domain constraints")
    solution_method: str = Field(default="formula application", description="Methodology required to solve")
    formulas: List[str] = Field(default_factory=list, description="Identified mathematical relationships")


class EvaluationResult(BaseModel):
    """Structured output returned by specialized evaluators."""
    passed: bool = Field(..., description="Whether candidate passed the evaluation criteria")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Evaluator confidence level")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Evaluated metric score")
    reasons: List[str] = Field(default_factory=list, description="Reasons for failure or notes")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Detailed sub-metrics")


class RAGKnowledgeContext(BaseModel):
    """Contextual knowledge payload retrieved from local RAG layer."""
    taxonomy: Dict[str, Any] = Field(default_factory=dict)
    topic_info: str = Field(default="")
    difficulty_rubric: Dict[str, Any] = Field(default_factory=dict)
    bloom_definitions: Dict[str, str] = Field(default_factory=dict)
    generation_patterns: List[str] = Field(default_factory=list)
    benchmark_examples: List[str] = Field(default_factory=list)


class FineTuningRecord(BaseModel):
    """Structured record stored for future local model fine-tuning dataset generation."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    seed_question: str
    blueprint: Dict[str, Any]
    generated_question: str
    accepted: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    evaluator_scores: Dict[str, float] = Field(default_factory=dict)
    answer_correctness: bool
    duplicate_score: float
    difficulty_score: float
    bloom_level: str
    timestamp: float = Field(default_factory=lambda: 0.0)




class GenerationRequest(BaseModel):
    """Official PS8 API Generation Request format."""
    seed_question: str = Field(..., description="The seed question to iterate upon")
    domain: str = Field(..., description="Selectable domain/subject area")
    count: int = Field(default=60, ge=1, le=200, description="Number of question variations to generate")
    target_difficulty: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional requested difficulty level")


class VariationOutput(BaseModel):
    """Official PS8 single question variation structure."""
    question: str = Field(..., description="Generated variation question text")
    answer_key: str = Field(..., description="Accompanying correct answer key and explanation")
    difficulty: float = Field(..., ge=0.0, le=1.0, description="Calculated difficulty score (0.0 to 1.0)")


class GenerationResponse(BaseModel):
    """Official PS8 API Generation Response format."""
    variations: List[VariationOutput] = Field(..., description="List of accepted question variations")
    duplicate_rate: float = Field(..., ge=0.0, le=1.0, description="Ratio of duplicate candidates detected during generation")


class DifficultyScore(BaseModel):
    """Evaluation output for difficulty analysis."""
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized difficulty score (0.0 to 1.0)")
    blooms_level: Optional[str] = Field(default=None, description="Bloom's Taxonomy classification")
    complexity_factors: Dict[str, Any] = Field(default_factory=dict, description="Detailed factors (e.g. math depth, reading level)")


class ValidationResult(BaseModel):
    """Validation output for a single generated variation candidate."""
    is_valid: bool = Field(..., description="Whether the variation passed all quality criteria")
    difficulty_score: DifficultyScore = Field(..., description="Evaluated difficulty metrics")
    is_difficulty_equivalent: bool = Field(..., description="Whether difficulty is within acceptable range of seed")
    answer_valid: bool = Field(..., description="Whether answer key is complete and valid")
    objective_valid: bool = Field(default=True, description="Whether variation preserves seed learning objective")
    reasons: List[str] = Field(default_factory=list, description="Reasons for rejection if invalid")


class DuplicateResult(BaseModel):
    """Duplicate detection result for a candidate variation."""
    is_duplicate: bool = Field(..., description="True if candidate exceeds similarity threshold")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Highest similarity score against existing dataset")
    matched_question_id: Optional[str] = Field(default=None, description="ID of matched duplicate question if any")


class QuestionVariation(BaseModel):
    """Internal domain model for a question variation lifecycle."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for variation")
    seed_question_id: Optional[str] = Field(default=None, description="ID or reference to seed question")
    question: str = Field(..., description="Variation question text")
    answer_key: str = Field(..., description="Generated answer key text")
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0, description="Calculated difficulty score")
    domain: Optional[str] = Field(default=None, description="Subject domain")
    learning_objective: Optional[str] = Field(default=None, description="Preserved learning objective")
    context_changes: Optional[str] = Field(default=None, description="Description of context/scenario variation")
    validation_result: Optional[ValidationResult] = Field(default=None, description="Quality validation result")
    duplicate_result: Optional[DuplicateResult] = Field(default=None, description="Duplicate check result")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Evaluator confidence score")


class AnswerKey(BaseModel):
    """Structured answer key component."""
    question_text: str = Field(..., description="Associated variation question text")
    answer_text: str = Field(..., description="Correct answer solution")
    explanation: Optional[str] = Field(default=None, description="Step-by-step reasoning or explanation")
    rubric_points: List[str] = Field(default_factory=list, description="Grading criteria points")


class ReviewItem(BaseModel):
    """Low-confidence candidate item queued for instructor review."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Review queue item ID")
    variation: QuestionVariation = Field(..., description="Question variation flagged for review")
    reason: str = Field(..., description="Reason for routing to review queue (e.g. low confidence, border duplicate)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Evaluator confidence score")
    status: str = Field(default="pending", description="Status: pending, approved, rejected")


class GenerationMetrics(BaseModel):
    """Execution telemetry and quality metrics for a generation run."""
    requested_count: int = Field(..., description="Number of variations requested")
    generated_count: int = Field(..., description="Total raw variations generated")
    accepted_count: int = Field(..., description="Variations passing evaluation")
    rejected_count: int = Field(..., description="Variations failing evaluation")
    regeneration_count: int = Field(default=0, description="Total regeneration attempts made")
    duplicate_count: int = Field(..., description="Variations flagged as duplicate")
    duplicate_rate: float = Field(..., ge=0.0, le=1.0, description="Ratio of duplicates among generated")
    low_confidence_count: int = Field(..., description="Variations sent to review queue")
    low_confidence_review_count: int = Field(default=0, description="Count of low-confidence items routed to review")
    objective_failure_count: int = Field(default=0, description="Variations failing learning objective validation")
    answer_failure_count: int = Field(default=0, description="Variations failing answer key validation")
    generation_time_seconds: float = Field(..., ge=0.0, description="Total pipeline execution time in seconds")
    total_wall_clock_time: float = Field(default=0.0, description="Total wall-clock pipeline time in seconds")
    llm_invocation_count: int = Field(default=0, description="Number of LLM API calls executed")
    average_llm_latency: float = Field(default=0.0, description="Average LLM call latency in seconds")
    objective_preservation_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Ratio of variations preserving objective")
    answer_key_correctness_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Ratio of variations with verified answer keys")
    difficulty_equivalence_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Ratio of variations matching baseline difficulty tolerance")
    variation_strategy_distribution: Dict[str, int] = Field(default_factory=dict, description="Counts of applied variation strategies")


class DomainInfo(BaseModel):
    """Subject domain metadata."""
    id: str
    name: str
    description: str


class DomainListResponse(BaseModel):
    """Response for GET /api/v1/domains."""
    domains: List[DomainInfo]
