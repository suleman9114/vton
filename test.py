import requests
import base64
from PIL import Image
import io
import json
import time

# Replace with your actual endpoint URL
API_URL = "https://m0w3ywx9kin4nrxb.us-east-1.aws.endpoints.huggingface.cloud"

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Submit the try-on request
def submit_tryon_request(human_image_path, garment_image_path, garment_description, category):
    human_img_base64 = encode_image(human_image_path)
    garment_img_base64 = encode_image(garment_image_path)
    
    payload = {
        "human_img_base64": human_img_base64,
        "garm_img_base64": garment_img_base64,
        "garment_des": garment_description,
        "category": category,
        "auto_mask": True,
        "denoise_steps": 30,
        "seed": 1
    }
    
    response = requests.post(f"{API_URL}/predict", headers=headers, json=payload)
    return response.json()

# Check job status
def check_job_status(job_id):
    response = requests.get(f"{API_URL}/job/{job_id}", headers=headers)
    return response.json()

# Example usage
human_image_path = "/home/devhouse/Downloads/8508152-6590197-image-m-74_1547477433411.jpg"
garment_image_path = "/home/devhouse/Downloads/c06a513905de33b7e59742454822a84a.jpg"
garment_description = "a shirt"
category = "upper" # Use "upper" for upper body garments, "lower" for pants, etc.

# Submit the request
result = submit_tryon_request(human_image_path, garment_image_path, garment_description, category)
print("Submitted job:", result)

job_id = result.get("job_id")
if job_id:
    # Poll for job completion
    while True:
        status = check_job_status(job_id)
        print("Job status:", status)
        
        if status.get("status") == "completed":
            # If result is base64 encoded image
            if status.get("result"):
                image_data = base64.b64decode(status["result"])
                image = Image.open(io.BytesIO(image_data))
                image.save("result.jpg")
                print("Image saved to result.jpg")
            break
        elif status.get("status") == "failed":
            print("Job failed:", status.get("error"))
            break
            
        time.sleep(5)  # Wait before checking again