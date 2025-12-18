from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from services.job_service import JOBS
import mimetypes
from pathlib import Path

router = APIRouter()


@router.get("/result/{job_id}")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "COMPLETED":
        raise HTTPException(status_code=409, detail="Job not completed")

    file_path = Path(job.result_path)

    media_type, _ = mimetypes.guess_type(file_path.name)
    media_type = media_type or "application/octet-stream"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name
    )
