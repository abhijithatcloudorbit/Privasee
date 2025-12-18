import time
import shutil
import threading
from pathlib import Path
from typing import Dict
from models.job import Job, JobStatus

JOBS: Dict[str, Job] = {}

PROCESSED_DIR = Path("storage/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def create_job(job_id: str, file_id: str):
    JOBS[job_id] = Job(
        job_id=job_id,
        file_id=file_id,
        status=JobStatus.PENDING,
        progress=0,
    )


def process_job(job_id: str, source_path: Path):
    job = JOBS[job_id]
    job.status = JobStatus.RUNNING

    for i in range(1, 6):
        time.sleep(1)
        job.progress = i * 20

    result_path = PROCESSED_DIR / source_path.name
    shutil.copy(source_path, result_path)

    job.status = JobStatus.COMPLETED
    job.result_path = str(result_path)
