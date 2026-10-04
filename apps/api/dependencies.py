"""
Dependency injection container for API routes.
Provides MasterAgent, services, repositories, RAG providers, and model providers (MockLLMProvider or LocalLLMProvider).
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from packages.common.config import settings
from packages.common.providers.mock_provider import MockLLMProvider, MockEmbeddingProvider, MockVectorStore
from packages.common.providers.local_provider import LocalLLMProvider
from packages.common.rag import DefaultKnowledgeContextProvider
from apps.api.database import get_db, engine, Base
from apps.api.repository import GenerationRepository

from apps.generator.parser import DefaultQuestionParser
from apps.generator.objective import DefaultLearningObjectiveAnalyzer
from apps.generator.answer import DefaultAnswerKeyGenerator
from apps.generator.variation import DefaultVariationGenerator
from apps.generator.seed_analyzer import DefaultSeedAnalyzer
from apps.generator.planner import DefaultVariationPlanner
from apps.generator.blueprint import DefaultBlueprintGenerator
from apps.generator.solver import DefaultAnswerSolver

from apps.evaluator.difficulty import DefaultDifficultyAnalyzer, DefaultDifficultyEquivalenceValidator
from apps.evaluator.duplicate import DefaultDuplicateDetector
from apps.evaluator.answer_validator import DefaultAnswerValidator
from apps.evaluator.objective_validator import DefaultLearningObjectiveConsistencyValidator
from apps.evaluator.bloom import DefaultBloomClassifier, DefaultBloomEvaluator
from apps.evaluator.specialized import SpecializedQualityGate

from apps.orchestrator.agent import MasterAgent
from apps.orchestrator.interfaces import OrchestratorInterface

# Initialize database schema tables
Base.metadata.create_all(bind=engine)

# Singleton instances for embeddings and vector index in dev/test environment
_embedding_provider = MockEmbeddingProvider(dimension=64)
_vector_store = MockVectorStore()


def get_llm_provider():
    """Factory creating appropriate LLMProvider based on environment configuration."""
    if settings.llm_provider.lower() == "local":
        return LocalLLMProvider()
    return MockLLMProvider()


def get_orchestrator() -> OrchestratorInterface:
    """Factory yielding configured MasterAgent instance."""
    llm = get_llm_provider()

    parser = DefaultQuestionParser(llm_provider=llm)
    objective_analyzer = DefaultLearningObjectiveAnalyzer(llm_provider=llm)
    seed_analyzer = DefaultSeedAnalyzer(parser=parser, objective_analyzer=objective_analyzer, llm_provider=llm)

    knowledge_provider = DefaultKnowledgeContextProvider()
    planner = DefaultVariationPlanner()
    blueprint_generator = DefaultBlueprintGenerator()
    solver = DefaultAnswerSolver()

    answer_generator = DefaultAnswerKeyGenerator(llm_provider=llm, solver=solver)
    generator = DefaultVariationGenerator(
        llm_provider=llm,
        answer_generator=answer_generator,
        blueprint_generator=blueprint_generator
    )

    diff_analyzer = DefaultDifficultyAnalyzer()
    equiv_validator = DefaultDifficultyEquivalenceValidator()
    dup_detector = DefaultDuplicateDetector(embedding_provider=_embedding_provider, vector_store=_vector_store)
    ans_validator = DefaultAnswerValidator()
    obj_validator = DefaultLearningObjectiveConsistencyValidator()
    bloom_evaluator = DefaultBloomEvaluator(classifier=DefaultBloomClassifier())

    quality_gate = SpecializedQualityGate(
        objective_validator=obj_validator,
        answer_validator=ans_validator,
        difficulty_analyzer=diff_analyzer,
        equivalence_validator=equiv_validator,
        duplicate_detector=dup_detector,
        bloom_evaluator=bloom_evaluator
    )

    agent = MasterAgent(
        seed_analyzer=seed_analyzer,
        knowledge_provider=knowledge_provider,
        planner=planner,
        generator=generator,
        answer_solver=solver,
        answer_generator=answer_generator,
        quality_gate=quality_gate,
        duplicate_detector=dup_detector,
        embedding_provider=_embedding_provider,
        vector_store=_vector_store
    )
    return agent


def get_repository(db: Session = Depends(get_db)) -> GenerationRepository:
    """Dependency yielding GenerationRepository instance bound to DB session."""
    return GenerationRepository(db)
