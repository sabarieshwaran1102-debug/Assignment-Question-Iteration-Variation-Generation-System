"""
Common exception hierarchy for the AMIGO platform.
"""

class AMIGOError(Exception):
    """Base exception for all AMIGO domain errors."""
    pass


class ValidationError(AMIGOError):
    """Raised when request or domain validation fails."""
    pass


class ProviderError(AMIGOError):
    """Raised when an external or mock model provider fails."""
    pass


class EvaluationError(AMIGOError):
    """Raised during evaluation pipeline failures."""
    pass


class OrchestrationError(AMIGOError):
    """Raised when workflow orchestration fails."""
    pass


class ResourceNotFoundError(AMIGOError):
    """Raised when requested resource or domain is not found."""
    pass
