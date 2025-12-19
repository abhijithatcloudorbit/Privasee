from fastapi import APIRouter, HTTPException
from services.job_service import JOBS
from storage.s3_client import get_s3_client
import os

router = APIRouter()

s3 = get_s3_client()
PROCESSED_BUCKET = os.getenv("S3_PROCESSED_BUCKET")


@router.get("/result/{job_id}")
def get_result(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "COMPLETED":
        raise HTTPException(status_code=409, detail="Job not completed")

    if not job.result_key:
        raise HTTPException(status_code=500, detail="Result not available")

    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": PROCESSED_BUCKET,
            "Key": job.result_key
        },
        ExpiresIn=600
    )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "download_url": presigned_url
    }
