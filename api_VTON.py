import argparse
import torch
import os
import io
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
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
import tempfile  # Add this import for temporary files

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
parser.add_argument("--ipv6", action="store_true", help="Enable IPv6 support")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")


# Class for tracking job status
class JobStatus(BaseModel):
    id: str
    status: str
    start_time: float
    result_path: Optional[str] = None
    error: Optional[str] = None

# Response models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class TryOnResponse(BaseModel):
    job_id: str
    message: str

class JobStatusResponse(BaseModel):
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

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
                 denoise_steps=30, seed=1, load_mode=None, jacket=False, hip_ratio=0.15):
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
            mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints, jacket=jacket, hip_ratio=hip_ratio)
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
                     auto_mask, denoise_steps, seed, load_mode, jacket, hip_ratio=0.15):
    try:
        processing_jobs[job_id].status = "processing"
        result_path, result_img = process_images(
            human_img_data, garm_img_data, garment_des, category, 
            auto_mask, denoise_steps, seed, load_mode, jacket, hip_ratio
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

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=pipe is not None
    )

@app.post("/tryon", response_model=TryOnResponse)
async def tryon(
    background_tasks: BackgroundTasks,
    human_img: UploadFile = File(...),
    garm_img: UploadFile = File(...),
    garment_des: str = Form(...),
    category: str = Form(...),
    auto_mask: bool = Form(True),
    denoise_steps: int = Form(30),
    seed: int = Form(1),
    jacket: bool = Form(False),  # Add new jacket parameter
    hip_ratio: float = Form(0.15)  # Add hip_ratio parameter
):
    # Read the uploaded files
    human_img_data = await human_img.read()
    garm_img_data = await garm_img.read()
    
    # Apply overlay to garment image if jacket is True
    if jacket:
        # Save garment image to a temporary file
        temp_garm_path = os.path.join("temp", f"garment_{uuid.uuid4()}.png")
        os.makedirs("temp", exist_ok=True)
        with open(temp_garm_path, "wb") as f:
            f.write(garm_img_data)
        
        # Path to the overlay image
        overlay_img_path = "PHOTO-2025-03-11-06-55-45-removebg-preview.png"
        
        # Output path for the modified garment image
        modified_garm_path = os.path.join("temp", f"modified_garment_{uuid.uuid4()}.png")
        
        # Import and use the hardcoded overlay function
        from hardcoded_overlay import apply_overlay_with_hardcoded_config, apply_white_background, auto_crop_image
        apply_overlay_with_hardcoded_config(temp_garm_path, overlay_img_path, modified_garm_path)
        
        # Auto-crop to remove extra space
        cropped_path = os.path.join("temp", f"modified_garment_cropped_{uuid.uuid4()}.png")
        auto_crop_image(modified_garm_path, cropped_path)
        
        # Apply white background to the cropped image
        white_bg_path = os.path.join("temp", f"modified_garment_white_bg_{uuid.uuid4()}.png")
        apply_white_background(cropped_path, white_bg_path)
        
        # Read the modified garment image with white background
        with open(white_bg_path, "rb") as f:
            garm_img_data = f.read()
        
        # Clean up temporary files
        try:
            os.remove(temp_garm_path)
            os.remove(modified_garm_path)
            os.remove(cropped_path)
            os.remove(white_bg_path)
        except:
            pass
    
    # Create a job ID
    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = JobStatus(
        id=job_id,
        status="queued",
        start_time=time.time()
    )
    
    # Use executor directly instead of trying to get event loop in a background task
    # This fixes the "There is no current event loop in thread" error
    background_tasks.add_task(
        executor.submit,
        process_tryon_task,
        job_id, human_img_data, garm_img_data, garment_des, category, 
        auto_mask, denoise_steps, seed, args.load_mode, jacket, hip_ratio
    )
    
    return TryOnResponse(
        job_id=job_id,
        message="Processing job started"
    )

@app.get("/predict/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    if job_id not in processing_jobs:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )
    
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
    
    # Hardcode host to IPv6 address
    host = "::"
    
    # Run the server
    print(f"Starting server on {host}:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
