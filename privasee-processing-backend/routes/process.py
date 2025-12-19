import uuid
import threading
import os
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query
from services.job_service import create_job, process_job

router = APIRouter()

RAW_BUCKET = os.getenv("S3_RAW_BUCKET")


def normalize_raw_key(raw_file_key: str) -> str:
    key = unquote(raw_file_key.strip())

    if RAW_BUCKET and key.startswith(f"{RAW_BUCKET}/"):
        key = key.replace(f"{RAW_BUCKET}/", "", 1)

    if not key.startswith("raw/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid raw_file_key: {key}"
        )

    return key


@router.post("/process")
def start_processing(
    file_id: str = Query(...),
    raw_file_key: str = Query(...)
):
    if not file_id:
        raise HTTPException(status_code=400, detail="file_id missing")

    if not raw_file_key:
        raise HTTPException(status_code=400, detail="raw_file_key missing")

    normalized_key = normalize_raw_key(raw_file_key)

    job_id = str(uuid.uuid4())
    create_job(job_id, file_id)

    thread = threading.Thread(
        target=process_job,
        args=(job_id, normalized_key),
        daemon=True
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "PENDING"
    }
