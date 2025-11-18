#script to upload folders to s3
#python3 upload_data.py <folder> [bucket]

import sys
import boto3
import os
from pathlib import Path


def upload_folder(folder_path, bucket_name, region='us-east-2'):
    s3_client = boto3.client('s3', region_name=region)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"No folder found at: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a directory")

    folder_path = Path(folder_path)
    upload_count = 0
    error_count = 0

    print(f"Uploading contents of {folder_path} to S3...")

    for file_path in folder_path.rglob('*'):
        if file_path.is_file():

            relative_path = file_path.relative_to(folder_path)
            
            if relative_path.parts[0] == 'img':
                #files in img subfolder go to raw/img/
                s3_key = f'raw/img/{relative_path.name}'
            else:
                # Files in root folder go to raw/
                s3_key = f'raw/{relative_path.name}'

            print(f"Uploading {relative_path} to s3://{bucket_name}/{s3_key}")

            try:
                s3_client.upload_file(
                    str(file_path),
                    bucket_name,
                    s3_key,
                    ExtraArgs={
                        'ServerSideEncryption': 'AES256',
                        'Metadata': {
                            'uploaded-by': 'upload_script',
                            'original-filename': file_path.name
                        }
                    }
                )
                upload_count += 1
            except Exception as e:
                print(f"Error uploading {relative_path}: {str(e)}")
                error_count += 1

    print(f"\nUpload complete: {upload_count} files uploaded, {error_count} errors")
    return error_count == 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Invalid format given, requires 2 arguments")
        sys.exit(1)
    else:
        data_file = sys.argv[1]
        bucket = sys.argv[2]

        succeeded = upload_folder(data_file, bucket)
        sys.exit(0 if succeeded else 1)