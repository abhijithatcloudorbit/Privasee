import uuid
from pathlib import Path
from fastapi import UploadFile

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_upload(file: UploadFile) -> str:
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return file_id, file_path
