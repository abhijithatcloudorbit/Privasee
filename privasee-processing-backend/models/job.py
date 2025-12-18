from enum import Enum
from pydantic import BaseModel
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Job(BaseModel):
    job_id: str
    file_id: str
    status: JobStatus
    progress: int = 0
    result_path: Optional[str] = None
    error: Optional[str] = None
