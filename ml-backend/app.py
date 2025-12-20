from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from face.facemodel import blur_faces_from_bytes
from license_plate.lp_model import blur_lp_from_bytes

from uuid import uuid4
import threading
import os
import traceback
import jwt

from supabase import create_client

# -----------------------------
# SUPABASE SETUP
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase environment variables not set")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

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
# IN-MEMORY STORAGE
# -----------------------------
FILES: dict[str, bytes] = {}
RESULTS: dict[str, bytes] = {}

# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------
os.makedirs("outputs/face", exist_ok=True)
os.makedirs("outputs/lp", exist_ok=True)

# -----------------------------
# AUTH HELPERS
# -----------------------------
def extract_user_id(authorization: str | None):
    if not authorization:
        return None

    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("sub")
    except Exception:
        return None

# -----------------------------
# UPLOAD FILE
# -----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_bytes = await file.read()
    file_id = str(uuid4())

    FILES[file_id] = file_bytes

    return {"file_id": file_id}

# -----------------------------
# BACKGROUND JOB
# -----------------------------
def run_job(job_id: str, file_id: str, mode: str):
    try:
        supabase.table("jobs").update({
            "status": "PROCESSING",
            "progress": 30
        }).eq("job_id", job_id).execute()

        img_bytes = FILES[file_id]

        if mode == "license_plate":
            result = blur_lp_from_bytes(img_bytes)
        elif mode == "face":
            result = blur_faces_from_bytes(img_bytes)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        RESULTS[job_id] = result

        supabase.table("jobs").update({
            "status": "COMPLETED",
            "progress": 100
        }).eq("job_id", job_id).execute()

    except Exception:
        traceback.print_exc()
        supabase.table("jobs").update({
            "status": "FAILED",
            "progress": 0
        }).eq("job_id", job_id).execute()

# -----------------------------
# START PROCESSING
# -----------------------------
@app.post("/process")
def start_processing(
    file_id: str = Query(...),
    mode: str = Query(...),
    authorization: str = Header(None)
):
    if file_id not in FILES:
        raise HTTPException(status_code=404, detail="File not found")

    if mode not in ("face", "license_plate"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    user_id = extract_user_id(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    job_id = str(uuid4())

    supabase.table("jobs").insert({
        "job_id": job_id,
        "user_id": user_id,
        "file_id": file_id,
        "mode": mode,
        "status": "QUEUED",
        "progress": 0
    }).execute()

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
    job = (
        supabase
        .table("jobs")
        .select("status, progress")
        .eq("job_id", job_id)
        .single()
        .execute()
    )

    if not job.data:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": job.data["status"],
        "progress": job.data["progress"],
    }

# -----------------------------
# GET RESULT
# -----------------------------
@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = (
        supabase
        .table("jobs")
        .select("status")
        .eq("job_id", job_id)
        .single()
        .execute()
    )

    if not job.data:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.data["status"] != "COMPLETED":
        raise HTTPException(status_code=400, detail="Job not completed")

    result_bytes = RESULTS.get(job_id)
    if not result_bytes:
        raise HTTPException(status_code=500, detail="Result not available")

    return Response(
        content=result_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": "inline; filename=processed.jpg"
        },
    )

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
