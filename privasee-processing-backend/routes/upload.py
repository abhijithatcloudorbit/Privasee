from fastapi import APIRouter, UploadFile, File
from services.file_service import save_upload

router = APIRouter()


@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_id, path = save_upload(file)

    return {
        "file_id": file_id,
        "filename": file.filename
    }
