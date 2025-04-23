import requests
import base64
import io
from PIL import Image
import os
import time
import json

API_URL = "https://m0w3ywx9kin4nrxb.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def try_on_garment(human_img_path, garment_img_path, garment_desc, category, timeout=300):
    # Read and encode images
    print(f"Loading human image from: {human_img_path}")
    with open(human_img_path, 'rb') as f:
        human_img_data = f.read()
        human_img_base64 = base64.b64encode(human_img_data).decode('utf-8')
    
    print(f"Loading garment image from: {garment_img_path}")
    with open(garment_img_path, 'rb') as f:
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
    
    print(f"Sending request to try on {garment_desc}...")
    print(f"Request URL: {API_URL}/predict")
    print(f"Request headers: {json.dumps(headers, indent=2)}")
    print(f"Payload structure (not showing encoded images):")
    payload_info = {k: v if k not in ["human_img_base64", "garm_img_base64"] else f"[BASE64 data, length: {len(v)} chars]" for k, v in payload.items()}
    print(json.dumps(payload_info, indent=2))
    
    start_time = time.time()
    
    # Send request
    try:
        print(f"Sending request to API with timeout of {timeout} seconds...")
        print("This may take a few minutes - the model processes images on the server side.")
        print("Press Ctrl+C to cancel if it takes too long.")
        
        try:
            response = requests.post(
                f"{API_URL}/predict", 
                headers=headers, 
                json=payload,
                timeout=timeout
            )
        except requests.exceptions.Timeout:
            print(f"Request timed out after {timeout} seconds. The model may still be processing your images.")
            print("You can check the results later or try again with different parameters.")
            return None
            
        print(f"Response received after {time.time() - start_time:.2f} seconds")
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {json.dumps(dict(response.headers), indent=2)}")
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response text: {response.text}")
            return None
        
        print("Parsing response JSON...")
        try:
            result = response.json()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Raw response: {response.text[:500]}...")
            return None
            
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        # Print response structure without the actual result image
        result_info = {k: v if k != "result" else f"[BASE64 data, length: {len(v) if v else 0} chars]" for k, v in result.items()}
        print(f"Response structure:")
        print(json.dumps(result_info, indent=2))
        
        if result.get('status') == 'success' and result.get('result'):
            # Decode and save the result image
            print("Decoding result image...")
            img_data = base64.b64decode(result['result'])
            img = Image.open(io.BytesIO(img_data))
            
            # Create output directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            # Generate output filename
            human_name = os.path.basename(human_img_path).split('.')[0]
            garment_name = os.path.basename(garment_img_path).split('.')[0]
            output_path = f"results/{human_name}_{garment_name}.png"
            
            print(f"Saving result image to {output_path}...")
            img.save(output_path)
            print(f"Result saved to {output_path}")
            return output_path
        else:
            print(f"Error: Unexpected response format: {result}")
            return None
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. The server may still be processing your request.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with example images
    human_img = "example/human/00121_00.jpg"
    garment_img = "example/cloth/09290_00.jpg"
    garment_desc = "blue t-shirt"
    category = "upper_body"  # Options: "upper_body", "lower_body", "dresses"
    
    print("=== Starting virtual try-on test ===")
    print("Note: This process typically takes 1-3 minutes to complete.")
    result_path = try_on_garment(human_img, garment_img, garment_desc, category)
    if result_path:
        print(f"Success! Result saved to: {result_path}")
    else:
        print("Failed to complete virtual try-on.")
    print("=== Test completed ===") 