#!/usr/bin/env python
import requests
import base64
import io
import argparse
import time
import os
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Test the IDM-VTON FastAPI')
    parser.add_argument('--human', type=str, required=True, help='Path to human image')
    parser.add_argument('--garment', type=str, required=True, help='Path to garment image')
    parser.add_argument('--description', type=str, required=True, help='Description of garment')
    parser.add_argument('--category', type=str, default='upper_body', 
                        choices=['upper_body', 'lower_body', 'dresses'], 
                        help='Category of garment')
    parser.add_argument('--host', type=str, default='http://localhost:7860', 
                        help='API host URL')
    parser.add_argument('--denoise-steps', type=int, default=30, 
                        help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=1, 
                        help='Random seed for generation')
    parser.add_argument('--output', type=str, default='result.png', 
                        help='Output image path')
    
    args = parser.parse_args()
    
    # Check if the API is up
    try:
        health_response = requests.get(f"{args.host}/health")
        health_data = health_response.json()
        if health_data['status'] != 'ok' or not health_data['model_loaded']:
            print("API is not ready. Please ensure the server is running and models are loaded.")
            return
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to the API at {args.host}. Please ensure the server is running.")
        return
    
    print("API is ready. Submitting try-on job...")
    
    # Read the image files
    try:
        with open(args.human, 'rb') as f:
            human_img = f.read()
        with open(args.garment, 'rb') as f:
            garment_img = f.read()
    except FileNotFoundError as e:
        print(f"Error reading image file: {e}")
        return
    
    # Submit the try-on job
    try:
        response = requests.post(
            f'{args.host}/tryon',
            files={
                'human_img': (os.path.basename(args.human), human_img),
                'garm_img': (os.path.basename(args.garment), garment_img),
            },
            data={
                'garment_des': args.description,
                'category': args.category,
                'auto_mask': 'true',
                'denoise_steps': str(args.denoise_steps),
                'seed': str(args.seed)
            }
        )
        
        if response.status_code != 200:
            print(f"Error submitting job: {response.text}")
            return
        
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"Job submitted with ID: {job_id}")
    except requests.exceptions.RequestException as e:
        print(f"Error submitting job: {e}")
        return
    
    # Poll for job status
    print("Waiting for job to complete...")
    poll_count = 0
    while True:
        try:
            response = requests.get(f'{args.host}/job/{job_id}')
            if response.status_code != 200:
                print(f"Error checking job status: {response.text}")
                return
            
            job_data = response.json()
            status = job_data['status']
            
            if status == 'completed':
                print("Try-on completed!")
                # Decode and save the image
                img_data = base64.b64decode(job_data['result'])
                img = Image.open(io.BytesIO(img_data))
                img.save(args.output)
                print(f"Result saved to {args.output}")
                break
            elif status == 'failed':
                print(f"Try-on failed: {job_data.get('error', 'Unknown error')}")
                break
            else:
                poll_count += 1
                if poll_count % 4 == 0:  # Show status every ~20 seconds
                    print(f"Job status: {status}, still waiting...")
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"Error checking job status: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main() 