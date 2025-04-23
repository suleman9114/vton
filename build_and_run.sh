#!/bin/bash

# Exit on any error
set -e

# Build the Docker image
echo "Building Docker image..."
docker build -t idm-vton-hf:latest -f Dockerfile.hf .

# Tag the image for Docker Hub
echo "Tagging Docker image..."
docker tag idm-vton-hf:latest yourusername/idm-vton-hf:latest

# Optional: Push to Docker Hub (uncomment if you want to push)
# echo "Pushing Docker image to Docker Hub..."
# docker push yourusername/idm-vton-hf:latest

# Run the container locally for testing
echo "Running Docker container locally..."
docker run --rm -it --gpus all -p 80:80 idm-vton-hf:latest

# Instructions for Hugging Face deployment
echo "
To deploy to Hugging Face Inference Endpoints:
1. Push the image to Docker Hub:
   docker push yourusername/idm-vton-hf:latest

2. Go to https://ui.endpoints.huggingface.co/
3. Create a new Dedicated endpoint
4. Select 'yisol/IDM-VTON' as the Model Repository
5. Under Advanced Configuration:
   - Set Task to 'Custom'
   - Set Container Type to 'Custom'
   - Set Container URL to 'yourusername/idm-vton-hf:latest'
   - Set Container port to '80'
   - Set Health route to '/health'
" 