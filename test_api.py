import requests
import base64
import io
import os
import time
import json
from PIL import Image

API_URL = "https://1c97-35-226-149-214.ngrok-free.app"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def check_health():
    """Check if the API is healthy and ready to accept requests."""
    print(f"Checking health at {API_URL}/health")
    try:
        response = requests.get(f"{API_URL}/health", headers=headers)
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Health check result: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Error checking health: {e}")
        return None

def test_predict():
    """Test the prediction endpoint with sample images."""
    # Define image paths
    human_img = "example/human/00121_00.jpg"
    garment_img = "example/cloth/09290_00.jpg"
    garment_desc = "blue t-shirt"
    category = "upper_body"  # Options: "upper_body", "lower_body", "dresses"
    
    # Read and encode images
    print(f"Loading human image from: {human_img}")
    with open(human_img, 'rb') as f:
        human_img_data = f.read()
        human_img_base64 = base64.b64encode(human_img_data).decode('utf-8')
    
    print(f"Loading garment image from: {garment_img}")
    with open(garment_img, 'rb') as f:
        garment_img_data = f.read()
        garment_img_base64 = base64.b64encode(garment_img_data).decode('utf-8')
    
    # Prepare payload
    payload = {
        "human_img_base64": human_img_base64,
        "garm_img_base64": garment_img_base64,
        "garment_des": garment_desc,
        "category": category,
        "auto_mask": True,
        "denoise_steps": 30,
        "seed": 1
    }
    
    print(f"Sending try-on request for {garment_desc}...")
    print(f"Request URL: {API_URL}/predict")
    
    # Send request
    start_time = time.time()
    try:
        print("Sending request. This may take several minutes...")
        response = requests.post(f"{API_URL}/predict", headers=headers, json=payload, timeout=300)
        
        print(f"Response received in {time.time() - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return None
        
        # Process response
        result = response.json()
        print(f"Response structure:")
        result_info = {k: v if k != "result" else f"[BASE64 data, length: {len(v) if v else 0} chars]" 
                     for k, v in result.items()}
        print(json.dumps(result_info, indent=2))
        
        if result.get("status") == "success" and result.get("result"):
            # Save the result image
            img_data = base64.b64decode(result["result"])
            img = Image.open(io.BytesIO(img_data))
            
            # Create output directory
            os.makedirs("results", exist_ok=True)
            
            # Generate output filename
            human_name = os.path.basename(human_img).split('.')[0]
            garment_name = os.path.basename(garment_img).split('.')[0]
            output_path = f"results/{human_name}_{garment_name}.png"
            
            img.save(output_path)
            print(f"Result saved to {output_path}")
            return output_path
        else:
            print(f"Error: Unexpected response format")
            return None
    except requests.exceptions.Timeout:
        print(f"Request timed out after {time.time() - start_time:.2f} seconds")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=== Testing API ===")
    health_result = check_health()
    if health_result and health_result.get("status") == "ok" and health_result.get("model_loaded"):
        print("\n=== API is healthy and ready ===")
        print("\n=== Testing prediction endpoint ===")
        result_path = test_predict()
        if result_path:
            print(f"\n=== Success! Result saved to {result_path} ===")
        else:
            print("\n=== Prediction failed ===")
    else:
        print("\n=== API health check failed ===") 