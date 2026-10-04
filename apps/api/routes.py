"""
FastAPI route definitions implementing official PS8 API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from packages.schemas.models import (
    GenerationRequest,
    GenerationResponse,
    DomainListResponse,
    DomainInfo,
)
from apps.orchestrator.interfaces import OrchestratorInterface
from apps.api.repository import GenerationRepository
from apps.api.dependencies import get_orchestrator, get_repository

router = APIRouter(prefix="/api/v1", tags=["PS8 Generation System"])

# Selectable subject domains (at least 5 required by PS8)
AVAILABLE_DOMAINS = [
    DomainInfo(id="cs", name="Computer Science", description="Algorithms, data structures, software architecture, systems"),
    DomainInfo(id="physics", name="Physics", description="Classical mechanics, electromagnetism, thermodynamics, quantum mechanics"),
    DomainInfo(id="math", name="Mathematics", description="Calculus, linear algebra, probability, differential equations"),
    DomainInfo(id="ee", name="Electrical Engineering", description="Circuit analysis, signal processing, digital logic, power systems"),
    DomainInfo(id="chem", name="Chemistry", description="Physical chemistry, stoichiometry, thermodynamics, reaction kinetics"),
]


@router.post("/generate", response_model=GenerationResponse, status_code=status.HTTP_200_OK)
def generate_variations(
    request: GenerationRequest,
    orchestrator: OrchestratorInterface = Depends(get_orchestrator),
    repository: GenerationRepository = Depends(get_repository)
):
    """
    Official PS8 endpoint: Generate N question variations from a seed question.
    """
    if not request.seed_question or not request.seed_question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="seed_question cannot be empty."
        )

    if not request.domain or not request.domain.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="domain cannot be empty."
        )

    try:
        response, metrics, review_items = orchestrator.orchestrate_generation(request)
        
        # API owns persistence: Save run record and generated variations to DB
        repository.save_run(
            request=request,
            response=response,
            metrics=metrics,
            reviews=review_items
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation pipeline failed: {str(e)}"
        )


@router.get("/domains", response_model=DomainListResponse, status_code=status.HTTP_200_OK)
def list_domains():
    """
    Official PS8 endpoint: Get list of selectable academic domains.
    """
    return DomainListResponse(domains=AVAILABLE_DOMAINS)


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Official PS8 endpoint: Health check endpoint.
    """
    return {
        "status": "healthy",
        "phase": "Phase 1 - Backend Foundation & Shared Contracts",
        "version": "1.0.0"
    }
