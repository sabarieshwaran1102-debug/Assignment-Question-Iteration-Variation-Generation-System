"""
Configuration settings for AMIGO backend.
Supports environment variables for model providers, endpoints, and thresholds.
"""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    """Application configuration settings."""
    llm_provider: str = os.getenv("AMIGO_LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "mock")
    local_model: str = os.getenv("AMIGO_LOCAL_MODEL") or os.getenv("LOCAL_MODEL", "qwen3:4b")
    local_base_url: str = os.getenv("AMIGO_LOCAL_BASE_URL") or os.getenv("LOCAL_BASE_URL", "http://localhost:11434")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Quality thresholds
    duplicate_threshold: float = float(os.getenv("DUPLICATE_THRESHOLD", "0.85"))
    equivalence_tolerance: float = float(os.getenv("EQUIVALENCE_TOLERANCE", "0.25"))
    max_regeneration_attempts: int = int(os.getenv("MAX_REGENERATION_ATTEMPTS", "5"))


settings = Settings()
