from fastapi import APIRouter, HTTPException
from services.job_service import JOBS

router = APIRouter()


@router.get("/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress
    }
