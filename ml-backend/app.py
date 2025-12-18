from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from face.facemodel import blur_faces_from_bytes
from license_plate.lp_model import blur_lp_from_bytes

from uuid import uuid4
import threading
import os

# -----------------------------
# APP SETUP
# -----------------------------
app = FastAPI(title="Privacy Shield Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# IN-MEMORY STORAGE (DEMO SAFE)
# -----------------------------
FILES = {}   # file_id -> bytes
JOBS = {}    # job_id -> status/progress/result

# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------
os.makedirs("outputs/face", exist_ok=True)
os.makedirs("outputs/lp", exist_ok=True)

# -----------------------------
# UPLOAD FILE
# -----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_bytes = await file.read()
    file_id = str(uuid4())

    FILES[file_id] = file_bytes

    return {"file_id": file_id}

# -----------------------------
# BACKGROUND JOB LOGIC
# -----------------------------
def run_job(job_id: str, file_id: str, mode: str):
    try:
        JOBS[job_id]["status"] = "PROCESSING"
        JOBS[job_id]["progress"] = 30

        img_bytes = FILES[file_id]

        if mode == "license_plate":
            result = blur_lp_from_bytes(img_bytes)
        else:
            result = blur_faces_from_bytes(img_bytes)

        JOBS[job_id]["progress"] = 90
        JOBS[job_id]["result"] = result
        JOBS[job_id]["status"] = "COMPLETED"
        JOBS[job_id]["progress"] = 100

    except Exception as e:
        JOBS[job_id]["status"] = "FAILED"
        JOBS[job_id]["error"] = str(e)

# -----------------------------
# START PROCESSING
# -----------------------------
@app.post("/process")
def start_processing(file_id: str, mode: str = "face"):
    if file_id not in FILES:
        raise HTTPException(status_code=404, detail="File not found")

    job_id = str(uuid4())

    JOBS[job_id] = {
        "status": "QUEUED",
        "progress": 0,
        "result": None,
    }

    thread = threading.Thread(
        target=run_job,
        args=(job_id, file_id, mode),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}

# -----------------------------
# JOB STATUS
# -----------------------------
@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": job["status"],
        "progress": job["progress"],
    }

# -----------------------------
# GET RESULT
# -----------------------------
@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job not completed")

    return Response(
        content=job["result"],
        media_type="image/jpeg",
        headers={
            "Content-Disposition": "attachment; filename=processed.jpg"
        },
    )

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
