from fastapi import APIRouter

from app.services.health_service import get_health_status

router = APIRouter(tags=["Health"])


@router.get("/health/ready")
def readiness_check():
    return get_health_status()