import time
import io
import os
from typing import Dict
from models.job import Job, JobStatus
from storage.s3_client import get_s3_client

JOBS: Dict[str, Job] = {}

s3 = get_s3_client()

RAW_BUCKET = os.getenv("S3_RAW_BUCKET")
PROCESSED_BUCKET = os.getenv("S3_PROCESSED_BUCKET")


def create_job(job_id: str, file_id: str):
    JOBS[job_id] = Job(
        job_id=job_id,
        file_id=file_id,
        status=JobStatus.PENDING,
        progress=0
    )


def process_job(job_id: str, raw_file_key: str):
    job = JOBS[job_id]
    job.status = JobStatus.RUNNING

    # 1️⃣ Download raw file
    raw_buffer = io.BytesIO()
    s3.download_fileobj(
        Bucket=RAW_BUCKET,
        Key=raw_file_key,
        Fileobj=raw_buffer
    )
    raw_buffer.seek(0)

    # 2️⃣ Simulate processing
    for i in range(1, 6):
        time.sleep(1)
        job.progress = i * 20

    # 3️⃣ Upload processed file
    processed_key = f"processed/{raw_file_key.split('/')[-1]}"
    raw_buffer.seek(0)

    s3.upload_fileobj(
        Fileobj=raw_buffer,
        Bucket=PROCESSED_BUCKET,
        Key=processed_key
    )

    # 4️⃣ Finalize job
    job.status = JobStatus.COMPLETED
    job.result_key = processed_key
