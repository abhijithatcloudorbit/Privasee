import os
from .s3_client import get_s3_client

s3 = get_s3_client()

RAW_BUCKET = os.getenv("S3_RAW_BUCKET")
PROCESSED_BUCKET = os.getenv("S3_PROCESSED_BUCKET")


def upload_raw_file(file_obj, object_name: str):
    s3.upload_fileobj(
        file_obj,
        RAW_BUCKET,
        object_name
    )
    return f"{RAW_BUCKET}/{object_name}"


def upload_processed_file(file_obj, object_name: str):
    s3.upload_fileobj(
        file_obj,
        PROCESSED_BUCKET,
        object_name
    )
    return f"{PROCESSED_BUCKET}/{object_name}"
