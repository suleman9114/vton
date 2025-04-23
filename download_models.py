import os
import requests
from pathlib import Path
import gdown
import shutil

# Create directories if they don't exist
Path("ckpt/humanparsing").mkdir(parents=True, exist_ok=True)
Path("ckpt/openpose/ckpts").mkdir(parents=True, exist_ok=True)
Path("ckpt/densepose").mkdir(parents=True, exist_ok=True)

# Main Google Drive folder URL
main_folder_url = "https://drive.google.com/drive/folders/1azl9usQhT4mgJ-baMhwGuhdsb1_7DINc?usp=sharing"

print("Downloading model files from Google Drive...")

# Download the folders
try:
    # Use gdown to download all files from the main folder and subfolders
    print("Downloading all model files. This may take some time...")
    gdown.download_folder(
        url=main_folder_url,
        output="temp_models",
        quiet=False,
        remaining_ok=True
    )
    
    # Move files to their respective directories
    print("Moving files to appropriate directories...")
    
    # Process densepose folder
    if os.path.exists("temp_models/densepose"):
        for file in os.listdir("temp_models/densepose"):
            src_path = os.path.join("temp_models/densepose", file)
            dst_path = os.path.join("ckpt/densepose", file)
            if os.path.isfile(src_path):
                print(f"Moving {file} to ckpt/densepose")
                shutil.copy(src_path, dst_path)
    
    # Process humanparsing folder
    if os.path.exists("temp_models/humanparsing"):
        for file in os.listdir("temp_models/humanparsing"):
            src_path = os.path.join("temp_models/humanparsing", file)
            dst_path = os.path.join("ckpt/humanparsing", file)
            if os.path.isfile(src_path):
                print(f"Moving {file} to ckpt/humanparsing")
                shutil.copy(src_path, dst_path)
    
    # Process openpose folder - needs to go into ckpt/openpose/ckpts
    if os.path.exists("temp_models/openpose"):
        for file in os.listdir("temp_models/openpose"):
            src_path = os.path.join("temp_models/openpose", file)
            dst_path = os.path.join("ckpt/openpose/ckpts", file)
            if os.path.isfile(src_path):
                print(f"Moving {file} to ckpt/openpose/ckpts")
                shutil.copy(src_path, dst_path)
    
    # Remove temporary directory
    print("Cleaning up temporary files...")
    shutil.rmtree("temp_models", ignore_errors=True)
    
    print("Model download completed!")
    
except Exception as e:
    print(f"Error during download: {str(e)}")
    print("If direct folder download fails, you may need to manually download the folders from:")
    print(main_folder_url)
    print("And place the files in the following structure:")
    print("- ckpt/densepose/")
    print("- ckpt/humanparsing/")
    print("- ckpt/openpose/ckpts/")