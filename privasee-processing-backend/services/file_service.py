import uuid
from fastapi import UploadFile
from storage.storage_service import upload_raw_file


def save_upload(file: UploadFile):
    """
    Save uploaded file to S3 RAW bucket (MinIO locally).
    Returns a logical file_id and S3 object key.
    """

    file_id = str(uuid.uuid4())
    object_name = f"raw/{file_id}_{file.filename}"

    s3_key = upload_raw_file(
        file_obj=file.file,
        object_name=object_name
    )

    return file_id, s3_key
