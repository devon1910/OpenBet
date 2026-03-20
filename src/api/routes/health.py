from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.get("/pipeline-status")
async def pipeline_status():
    """Public endpoint — returns last pipeline run time and result from DB."""
    from src.workers.scheduler import get_pipeline_status
    return await get_pipeline_status()
