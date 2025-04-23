#!/usr/bin/env python
import requests
import base64
import io
import argparse
import time
import os
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Test the IDM-VTON Hugging Face Inference Endpoint')
    parser.add_argument('--human', type=str, required=True, help='Path to human image')
    parser.add_argument('--garment', type=str, required=True, help='Path to garment image')
    parser.add_argument('--description', type=str, required=True, help='Description of garment')
    parser.add_argument('--category', type=str, default='upper_body', 
                        choices=['upper_body', 'lower_body', 'dresses'], 
                        help='Category of garment')
    parser.add_argument('--endpoint', type=str, required=True, 
                        help='Hugging Face Inference Endpoint URL')
    parser.add_argument('--denoise-steps', type=int, default=30, 
                        help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=1, 
                        help='Random seed for generation')
    parser.add_argument('--output', type=str, default='result.png', 
                        help='Output image path')
    
    args = parser.parse_args()
    
    # Check if the endpoint URL is properly formatted
    if not args.endpoint.startswith("http"):
        print(f"Endpoint URL should start with http:// or https://")
        return
    
    # Check if the API is up by calling the health endpoint
    try:
        health_response = requests.get(f"{args.endpoint}/health")
        if health_response.status_code != 200:
            print(f"Health check failed with status code {health_response.status_code}")
            print(f"Response: {health_response.text}")
            return
        
        health_data = health_response.json()
        if health_data.get('status') != 'ok' or not health_data.get('model_loaded', False):
            print("API is not ready. Models are not loaded.")
            return
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to the endpoint: {e}")
        return
    
    print("API is ready. Submitting try-on job...")
    
    # Read and encode the image files to base64
    try:
        with open(args.human, 'rb') as f:
            human_img_data = f.read()
            human_img_base64 = base64.b64encode(human_img_data).decode('utf-8')
        
        with open(args.garment, 'rb') as f:
            garment_img_data = f.read()
            garment_img_base64 = base64.b64encode(garment_img_data).decode('utf-8')
    except FileNotFoundError as e:
        print(f"Error reading image file: {e}")
        return
    
    # Prepare the request payload
    payload = {
        "human_img_base64": human_img_base64,
        "garm_img_base64": garment_img_base64,
        "garment_des": args.description,
        "category": args.category,
        "auto_mask": True,
        "denoise_steps": args.denoise_steps,
        "seed": args.seed
    }
    
    # Send the prediction request
    print("Sending request to the model...")
    start_time = time.time()
    
    try:
        response = requests.post(f"{args.endpoint}/predict", json=payload)
        
        # Check for successful response
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        result = response.json()
        print(f"Processing completed in {time.time() - start_time:.2f} seconds")
        
        if result.get('status') == 'success' and result.get('result'):
            # Decode the base64 image and save it
            img_data = base64.b64decode(result['result'])
            img = Image.open(io.BytesIO(img_data))
            img.save(args.output)
            print(f"Result image saved to {args.output}")
        else:
            print(f"Error: Unexpected response format: {result}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
    except Exception as e:
        print(f"Error processing response: {e}")

if __name__ == "__main__":
    main() 