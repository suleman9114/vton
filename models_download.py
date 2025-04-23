import os
import requests
from pathlib import Path
import gdown
import shutil

# Get current directory
current_dir = os.getcwd()
# Define the target base directory - changing to local ckpt
target_base_dir = os.path.join(current_dir, "ckpt")

# Create directories if they don't exist
Path(os.path.join(target_base_dir, "humanparsing")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(target_base_dir, "openpose/ckpts")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(target_base_dir, "densepose")).mkdir(parents=True, exist_ok=True)

# Main Google Drive folder URL
main_folder_url = "https://drive.google.com/drive/folders/1azl9usQhT4mgJ-baMhwGuhdsb1_7DINc?usp=sharing"

print("Downloading model files from Google Drive...")

# Download the folders
try:
    # Use gdown to download all files from the main folder and subfolders
    print("Downloading all model files. This may take some time...")
    temp_dir = os.path.join(current_dir, "temp_models")
    gdown.download_folder(
        url=main_folder_url,
        output=temp_dir,
        quiet=False,
        remaining_ok=True
    )
    
    # Move files to their respective directories
    print("Moving files to appropriate directories...")
    
    # Process densepose folder
    if os.path.exists(os.path.join(temp_dir, "densepose")):
        for file in os.listdir(os.path.join(temp_dir, "densepose")):
            src_path = os.path.join(temp_dir, "densepose", file)
            dst_path = os.path.join(target_base_dir, "densepose", file)
            if os.path.isfile(src_path):
                print(f"Moving {file} to {dst_path}")
                shutil.copy2(src_path, dst_path)
    
    # Process humanparsing folder
    if os.path.exists(os.path.join(temp_dir, "humanparsing")):
        for file in os.listdir(os.path.join(temp_dir, "humanparsing")):
            src_path = os.path.join(temp_dir, "humanparsing", file)
            dst_path = os.path.join(target_base_dir, "humanparsing", file)
            if os.path.isfile(src_path):
                print(f"Moving {file} to {dst_path}")
                shutil.copy2(src_path, dst_path)
    
    # Process openpose folder - check for ckpts subfolder first
    openpose_dir = os.path.join(temp_dir, "openpose")
    openpose_ckpts_dir = os.path.join(openpose_dir, "ckpts")
    target_openpose_dir = os.path.join(target_base_dir, "openpose/ckpts")
    
    if os.path.exists(openpose_dir):
        if os.path.exists(openpose_ckpts_dir) and os.path.isdir(openpose_ckpts_dir):
            # If "ckpts" subfolder exists, copy from there
            for file in os.listdir(openpose_ckpts_dir):
                src_path = os.path.join(openpose_ckpts_dir, file)
                dst_path = os.path.join(target_openpose_dir, file)
                if os.path.isfile(src_path):
                    print(f"Moving {file} from openpose/ckpts to {dst_path}")
                    shutil.copy2(src_path, dst_path)
        else:
            # Otherwise, copy files directly from openpose folder
            for file in os.listdir(openpose_dir):
                src_path = os.path.join(openpose_dir, file)
                dst_path = os.path.join(target_openpose_dir, file)
                if os.path.isfile(src_path):
                    print(f"Moving {file} from openpose to {dst_path}")
                    shutil.copy2(src_path, dst_path)
    
    # Remove temporary directory
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("Model download completed!")
    print(f"Files have been placed in {target_base_dir}")
    
except Exception as e:
    print(f"Error during download: {str(e)}")
    print("If direct folder download fails, you may need to manually download the folders from:")
    print(main_folder_url)
    print("And place the files in the following structure:")
    print(f"- {target_base_dir}/densepose/")
    print(f"- {target_base_dir}/humanparsing/")
    print(f"- {target_base_dir}/openpose/ckpts/")