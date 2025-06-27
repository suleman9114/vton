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
    socat && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment with Python 3.10
RUN conda create -n vton python=3.10 -y
SHELL ["/bin/bash", "-c"]

# Copy the entire project directory
COPY . /app/

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

# Create necessary directories
RUN mkdir -p /app/outputs /app/temp

# Expose the port that the app runs on
EXPOSE 8080

# Set the default command
CMD ["bash", "-c", "source activate vton && python api_VTON.py --host :: --port 8080 --load_mode 8bit"]
