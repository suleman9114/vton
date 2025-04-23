# IDM-VTON Hugging Face Inference Endpoint

This repository contains the code for deploying the [IDM-VTON virtual try-on model](https://github.com/yisol/IDM-VTON) to Hugging Face Inference Endpoints.

## Overview

The IDM-VTON model allows you to virtually try on garments. This deployment provides a REST API with two endpoints:

1. `/health` - Health check endpoint that returns status of the model
2. `/predict` - Endpoint for virtual try-on requests

## Deployment to Hugging Face Inference Endpoints

### Building the Docker Image

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/IDM-VTON.git
   cd IDM-VTON
   ```

2. Build the Docker image:
   ```bash
   docker build -t idm-vton-hf:latest .
   ```

3. Push the image to a container registry (e.g., Docker Hub):
   ```bash
   docker tag idm-vton-hf:latest yourusername/idm-vton-hf:latest
   docker push yourusername/idm-vton-hf:latest
   ```

### Deploying on Hugging Face Inference Endpoints

1. Go to [Hugging Face Inference Endpoints](https://ui.endpoints.huggingface.co/)
2. Click on "Dedicated" tab and then "New"
3. Fill out the deployment form:
   - **Model Repository**: `yisol/IDM-VTON` (or your own fork)
   - **Endpoint Name**: Your choice
   - **Choose Hardware**: Select appropriate hardware (recommend at least 16GB VRAM)
   - Under **Advanced Configuration**:
     - **Task**: Set to "Custom"
     - **Container Type**: Select "Custom"
     - **Container URL**: Enter your Docker image URL (e.g., `yourusername/idm-vton-hf:latest`)
     - **Container port**: `80`
     - **Health route**: `/health`
4. Click "Create Endpoint" and wait for deployment to complete

## Using the API

### API Endpoints

#### Health Check

```
GET /health
```

Checks if the model is loaded and ready to accept requests.

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### Virtual Try-On

```
POST /predict
```

Performs virtual try-on with a garment image on a human image.

**Request Body (JSON)**:
```json
{
  "human_img_base64": "base64_encoded_human_image",
  "garm_img_base64": "base64_encoded_garment_image",
  "garment_des": "blue denim jacket",
  "category": "upper_body",
  "auto_mask": true,
  "denoise_steps": 30,
  "seed": 1
}
```

**Parameters**:
- `human_img_base64`: Base64-encoded human image (required)
- `garm_img_base64`: Base64-encoded garment image (required)
- `garment_des`: Description of the garment (required)
- `category`: Category of the garment - "upper_body", "lower_body", or "dresses" (required)
- `auto_mask`: Whether to use auto-masking (optional, default: true)
- `denoise_steps`: Number of denoising steps (optional, default: 30)
- `seed`: Random seed for generation (optional, default: 1)

**Response**:
```json
{
  "status": "success",
  "result": "base64_encoded_result_image"
}
```

### Example Usage with Python

You can use the provided `client_hf.py` script to test the API:

```bash
python client_hf.py \
  --human /path/to/human.jpg \
  --garment /path/to/garment.jpg \
  --description "blue denim jacket" \
  --category upper_body \
  --endpoint https://your-endpoint-url.endpoints.huggingface.cloud \
  --output result.png
```

Or you can write your own client:

```python
import requests
import base64
from PIL import Image
import io

# Read and encode images
with open('human.jpg', 'rb') as f:
    human_img_base64 = base64.b64encode(f.read()).decode('utf-8')

with open('garment.jpg', 'rb') as f:
    garm_img_base64 = base64.b64encode(f.read()).decode('utf-8')

# Prepare request
endpoint_url = "https://your-endpoint-url.endpoints.huggingface.cloud"
payload = {
    "human_img_base64": human_img_base64,
    "garm_img_base64": garm_img_base64,
    "garment_des": "blue denim jacket",
    "category": "upper_body",
    "auto_mask": True,
    "denoise_steps": 30,
    "seed": 1
}

# Make request
response = requests.post(f"{endpoint_url}/predict", json=payload)
result = response.json()

# Process response
if result["status"] == "success":
    img_data = base64.b64decode(result["result"])
    img = Image.open(io.BytesIO(img_data))
    img.save("result.png")
    print("Result saved to result.png")
else:
    print(f"Error: {result}")
```

## Notes

- The API processes requests in a synchronous manner: the `/predict` endpoint will not return until processing is complete
- The model is quite resource-intensive and may take 1-3 minutes to process a single image
- For high-traffic applications, you may want to adjust the number of workers in `api_VTON_hf.py` 