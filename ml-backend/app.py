from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from face.facemodel import blur_faces_from_bytes
from license_plate.lp_model import blur_lp_from_bytes

app = FastAPI(title="Privacy Shield Backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/blur/face")
async def blur_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    img_bytes = await file.read()
    output = blur_faces_from_bytes(img_bytes)
    return Response(content=output, media_type="image/jpeg")


@app.post("/api/blur/license-plate")
async def blur_lp(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    img_bytes = await file.read()
    output = blur_lp_from_bytes(img_bytes)
    return Response(content=output, media_type="image/jpeg")


@app.get("/health")
def health():
    return {"status": "ok"}

import os

os.makedirs("outputs/face", exist_ok=True)
os.makedirs("outputs/lp", exist_ok=True)
