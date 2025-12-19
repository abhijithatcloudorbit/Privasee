from fastapi import APIRouter, UploadFile, File
from services.file_service import save_upload

router = APIRouter()


@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_id, raw_file_key = save_upload(file)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "raw_file_key": raw_file_key
    }
