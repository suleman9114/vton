import argparse
import torch
import os
import io
import base64
from PIL import Image
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import time
import uvicorn
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)
from diffusers import AutoencoderKL
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import uuid

# Create FastAPI app
app = FastAPI(title="IDM-VTON API", description="Virtual Try-on API with Stable Diffusion")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
unet = None
pipe = None
UNet_Encoder = None
ENABLE_CPU_OFFLOAD = False
need_restart_cpu_offloading = False
processing_jobs = {}
executor = ThreadPoolExecutor(max_workers=1)  # Only allow one job at a time

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for optimization memory consumption")
parser.add_argument("--fixed_vae", action="store_true", default=True, help="Use fixed vae for FP16.")
parser.add_argument("--port", type=int, default=80, help="Port to run the API on")

# Classes for request and response
class JobStatus(BaseModel):
    id: str
    status: str
    start_time: float
    result_path: Optional[str] = None
    error: Optional[str] = None

class TryOnRequest(BaseModel):
    human_img_base64: str
    garm_img_base64: str
    garment_des: str
    category: str
    auto_mask: bool = True
    denoise_steps: int = 30
    seed: int = 1

class TryOnResponse(BaseModel):
    job_id: str
    message: str

class JobStatusResponse(BaseModel):
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Initialize models
def initialize_models(load_mode, fixed_vae):
    global unet, pipe, UNet_Encoder
    
    dtype = torch.float16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_id = 'yisol/IDM-VTON'
    vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'
    
    dtypeQuantize = dtype
    if load_mode in ('4bit', '8bit'):
        dtypeQuantize = torch.float8_e4m3fn
    
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.allow_tf32 = False
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtypeQuantize,
    )
    if load_mode == '4bit':
        quantize_4bit(unet)
    
    unet.requires_grad_(False)
    
    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    if load_mode == '4bit':
        quantize_4bit(image_encoder)
    
    # Load VAE
    if fixed_vae:
        vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
    else:
        vae = AutoencoderKL.from_pretrained(model_id,
                                        subfolder="vae",
                                        torch_dtype=dtype)
    
    # Load UNet Encoder
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        model_id,
        subfolder="unet_encoder",
        torch_dtype=dtypeQuantize,
    )
    
    if load_mode == '4bit':
        quantize_4bit(UNet_Encoder)
    
    UNet_Encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Create pipeline
    pipe_param = {
        'pretrained_model_name_or_path': model_id,
        'unet': unet,
        'torch_dtype': dtype,
        'vae': vae,
        'image_encoder': image_encoder,
        'feature_extractor': CLIPImageProcessor(),
    }
    
    pipe = TryonPipeline.from_pretrained(**pipe_param).to(device)
    pipe.unet_encoder = UNet_Encoder
    pipe.unet_encoder.to(pipe.unet.device)
    
    if load_mode == '4bit':
        if pipe.text_encoder is not None:
            quantize_4bit(pipe.text_encoder)
        if pipe.text_encoder_2 is not None:
            quantize_4bit(pipe.text_encoder_2)
    
    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    
    return pipe

# Process image function
def process_images(human_img_data, garm_img_data, garment_des, category, auto_mask=True, 
                 denoise_steps=30, seed=1, load_mode=None):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading
    
    try:
        # Convert base64 to PIL images
        human_img = Image.open(io.BytesIO(human_img_data)).convert("RGB").resize((768, 1024))
        garm_img = Image.open(io.BytesIO(garm_img_data)).convert("RGB").resize((768, 1024))
        
        torch_gc()
        parsing_model = Parsing(0)
        openpose_model = OpenPose(0)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        openpose_model.preprocessor.body_estimation.model.to(device)
        
        tensor_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        if need_restart_cpu_offloading:
            restart_cpu_offload(pipe, load_mode)
        elif ENABLE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        
        # Generate mask
        if auto_mask:
            keypoints = openpose_model(human_img.resize((384, 512)))
            model_parse, _ = parsing_model(human_img.resize((384, 512)))
            mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
            mask = mask.resize((768, 1024))
        else:
            # Default to full mask if auto_mask is disabled
            mask = Image.new('L', (768, 1024), 0)
        
        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)
        
        # Prepare pose image
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        
        args_apply = apply_net.create_argument_parser().parse_args((
            'show', 
            './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
            './ckpt/densepose/model_final_162be9.pkl', 
            'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'
        ))
        pose_img = args_apply.func(args_apply, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))
        
        # Move encoders to device
        if pipe.text_encoder is not None:
            pipe.text_encoder.to(device)
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to(device)
        
        # Run inference
        dtype = torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                with torch.no_grad():
                    prompt = "model is wearing " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    with torch.inference_mode():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )
                        prompt = "a photo of " + garment_des
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                        
                        if not isinstance(prompt, List):
                            prompt = [prompt] * 1
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * 1
                            
                        with torch.inference_mode():
                            (
                                prompt_embeds_c,
                                _,
                                _,
                                _,
                            ) = pipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                negative_prompt=negative_prompt,
                            )
                            
                        pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, dtype)
                        garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, dtype)
                        
                        generator = torch.Generator(device).manual_seed(seed) if seed != -1 else None
                        
                        images = pipe(
                            prompt_embeds=prompt_embeds.to(device, dtype),
                            negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength=1.0,
                            pose_img=pose_img_tensor.to(device, dtype),
                            text_embeds_cloth=prompt_embeds_c.to(device, dtype),
                            cloth=garm_tensor.to(device, dtype),
                            mask_image=mask,
                            image=human_img,
                            height=1024,
                            width=768,
                            ip_adapter_image=garm_img.resize((768, 1024)),
                            guidance_scale=2.0,
                            dtype=dtype,
                            device=device,
                        )[0]
        
        # Save and return result
        os.makedirs("outputs", exist_ok=True)
        result_filename = f"img_{uuid.uuid4()}.png"
        result_path = os.path.join("outputs", result_filename)
        images[0].save(result_path)
        
        # Convert to base64 for API response
        buffered = io.BytesIO()
        images[0].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return result_path, img_str
    
    except Exception as e:
        import traceback
        error_msg = f"Error in processing: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return None, error_msg

# Process try-on request in background
def process_tryon_task(job_id, human_img_data, garm_img_data, garment_des, category, 
                     auto_mask, denoise_steps, seed, load_mode):
    try:
        processing_jobs[job_id].status = "processing"
        result_path, result_img = process_images(
            human_img_data, garm_img_data, garment_des, category, 
            auto_mask, denoise_steps, seed, load_mode
        )
        
        if result_path:
            processing_jobs[job_id].status = "completed"
            processing_jobs[job_id].result_path = result_img  # Store base64 image
        else:
            processing_jobs[job_id].status = "failed"
            processing_jobs[job_id].error = result_img  # Store error message
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].error = str(e)

# API endpoints for Hugging Face Inference Endpoints

@app.get("/health")
async def health():
    """Health check endpoint for Hugging Face Inference Endpoints"""
    return HealthResponse(
        status="ok",
        model_loaded=pipe is not None
    )

@app.post("/predict")
async def predict(request: TryOnRequest, background_tasks: BackgroundTasks):
    """
    Prediction endpoint for Hugging Face Inference Endpoints
    Accepts base64 encoded images for human and garment
    """
    try:
        # Decode base64 images
        try:
            human_img_data = base64.b64decode(request.human_img_base64)
            garm_img_data = base64.b64decode(request.garm_img_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        # Create a job ID
        job_id = str(uuid.uuid4())
        processing_jobs[job_id] = JobStatus(
            id=job_id,
            status="queued",
            start_time=time.time()
        )
        
        # Process request in background
        background_tasks.add_task(
            executor.submit,
            process_tryon_task,
            job_id, human_img_data, garm_img_data, request.garment_des, request.category,
            request.auto_mask, request.denoise_steps, request.seed, args.load_mode
        )
        
        # For HF Inference Endpoints, we'll process synchronously to return result immediately
        max_wait_time = 300  # 5 minutes max wait time
        wait_interval = 1  # Check every second
        total_wait_time = 0
        
        while total_wait_time < max_wait_time:
            job = processing_jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            
            if job.status == "completed":
                return {"status": "success", "result": job.result_path}
            elif job.status == "failed":
                raise HTTPException(status_code=500, detail=job.error or "Processing failed")
            
            # Wait and check again
            await asyncio.sleep(wait_interval)
            total_wait_time += wait_interval
        
        # If we're here, the job took too long
        raise HTTPException(status_code=504, detail="Processing timeout")
        
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=str(e))
        raise

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job.status == "completed":
        return JobStatusResponse(
            status=job.status,
            result=job.result_path
        )
    elif job.status == "failed":
        return JobStatusResponse(
            status=job.status,
            error=job.error
        )
    else:
        return JobStatusResponse(
            status=job.status
        )

# Main function to run the app
if __name__ == "__main__":
    args = parser.parse_args()
    ENABLE_CPU_OFFLOAD = args.lowvram
    
    # Initialize the model
    print("Initializing models...")
    initialize_models(args.load_mode, args.fixed_vae)
    print("Models initialized successfully!")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=args.port) 