import boto3
import os
from datetime import datetime
import time

# AWS Config
BUCKET_NAME = "rapidevpcb"
S3_BASE_PATH = "data"

# List of folders to upload
FOLDERS_TO_UPLOAD = [
    "/home/osama/PCB_Inspection/App/results",
    "/home/osama/PCB_Inspection/App/Test_Images",
    "/home/osama/PCB_Inspection/App/saved",
]

# Create S3 client
s3 = boto3.client("s3")


def upload_folder_contents(folder_path):
    folder_name = os.path.basename(folder_path.rstrip("/"))
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    for root, _, files in os.walk(folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, folder_path)
            s3_key = f"{S3_BASE_PATH}/{folder_name}/{timestamp}/{relative_path.replace(os.sep, '/')}"
            try:
                s3.upload_file(local_file_path, BUCKET_NAME, s3_key)
                print(f"Uploaded: {local_file_path} → s3://{BUCKET_NAME}/{s3_key}")
            except Exception as e:
                print(f"Failed to upload {local_file_path}: {e}")


if __name__ == "__main__":
    while True:
        for folder in FOLDERS_TO_UPLOAD:
            upload_folder_contents(folder)
        time.sleep(3600)  # Wait for 1 hour
