from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.get("/pipeline-status")
async def get_pipeline_status():
    """Public endpoint — returns last pipeline run time and result."""
    from src.workers.scheduler import pipeline_status
    return pipeline_status
