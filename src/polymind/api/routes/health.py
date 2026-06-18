"""Health check endpoint."""

from fastapi import APIRouter

from polymind import __version__
from polymind.api.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", version=__version__, service="polymind")
