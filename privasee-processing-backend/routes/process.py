import uuid
import threading
from fastapi import APIRouter, HTTPException
from pathlib import Path
from services.job_service import create_job, process_job, JOBS

UPLOAD_DIR = Path("storage/uploads")
router = APIRouter()


@router.post("/process")
def start_processing(file_id: str):
    matching_files = list(UPLOAD_DIR.glob(f"{file_id}_*"))
    if not matching_files:
        raise HTTPException(status_code=404, detail="File not found")

    source_path = matching_files[0]

    job_id = str(uuid.uuid4())
    create_job(job_id, file_id)

    thread = threading.Thread(
        target=process_job,
        args=(job_id, source_path),
        daemon=True
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "PENDING"
    }
