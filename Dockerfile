FROM continuumio/miniconda3

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgoogle-perftools-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    socat \  # Add socat for IPv6 routing
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with Python 3.10
RUN conda create -n vton python=3.10 -y
SHELL ["/bin/bash", "-c"]

# Clone repository
RUN git clone https://github.com/suleman9114/vton.git .

# Install base requirements
RUN source activate vton && pip install -r requirements.txt

# Install additional dependencies
RUN source activate vton && \
    pip install gradio huggingface_hub hf_transfer flask pyngrok && \
    rm -rf /opt/conda/envs/vton/lib/python3.10/site-packages/numpy* && \
    pip install numpy==1.26.4 && \
    pip install bitsandbytes==0.43.0 accelerate==0.30.1 peft==0.11.1 --upgrade && \
    pip install pydantic==2.10.6 && \
    pip install torchvision==0.19.1 xformers --extra-index-url https://download.pytorch.org/whl/cu121

# Set environment variable for HuggingFace transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# Make Python print immediately without buffering
ENV PYTHONUNBUFFERED=1
# Force flush for stdout
ENV PYTHONFAULTHANDLER=1

ENV TRANSFORMERS_CACHE=/.cache/huggingface/hub

ENV MPLCONFIGDIR=/.config/matplotlib

RUN mkdir -p /app/outputs && chmod -R 777 /app/outputs

# Download model files
RUN source activate vton && python -u models_download.py

# Expose the port the app runs on
EXPOSE 80

# Create huggingface cache directories with proper permissions
RUN mkdir -p /.cache/huggingface/hub && chmod -R 777 /.cache
RUN mkdir -p /.config/matplotlib && chmod -R 777 /.config

# Make sure CUDA is visible to the container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create script to start app with IPv6 support
RUN echo '#!/bin/bash\n\
source activate vton\n\
# Start socat to forward IPv6 traffic to IPv4\n\
socat TCP6-LISTEN:80,fork TCP4:0.0.0.0:8080 &\n\
# Run the API on IPv4 address 0.0.0.0 port 8080\n\
python -u api_VTON.py --load_mode 8bit --port 8080\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the application with IPv6 support
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/app/start.sh"]
