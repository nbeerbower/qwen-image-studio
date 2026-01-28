from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal
from enum import Enum
import asyncio
import uuid
import os
import time
from datetime import datetime
import torch
from PIL import Image
from queue import Queue
import threading
import io
import json
import argparse
import glob as globmodule
from diffusers import DiffusionPipeline

app = FastAPI(title="Qwen Image Studio", version="1.0.0")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Qwen Image Studio Server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
parser.add_argument("--edit-model", type=str, default="2511", choices=["original", "2509", "2511"],
                    help="Which edit model to use: 'original', '2509', or '2511' (default: 2511)")
parser.add_argument("--generation-model", type=str, default="2512", choices=["original", "2512"],
                    help="Which generation model to use: 'original' or '2512' (default: 2512)")
parser.add_argument("--quantize", action="store_true",
                    help="Enable 4-bit quantization for edit model (only for original model)")
parser.add_argument("--cpu-offload", action="store_true", default=True,
                    help="Enable CPU offloading (default: True)")
parser.add_argument("--max-pixels", type=int, default=1048576,
                    help="Maximum pixels for image editing (default: 1048576)")
parser.add_argument("--disable-generation", action="store_true",
                    help="Disable image generation pipeline")
parser.add_argument("--disable-edit", action="store_true",
                    help="Disable image editing pipeline")
parser.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cpu"],
                    help="Device to use (default: auto)")
parser.add_argument("--pipeline-swap", action="store_true",
                    help="Keep pipelines in CPU memory and swap to GPU only when needed (saves VRAM)")
parser.add_argument("--keep-in-vram", type=str, default=None,
                    choices=["generation", "edit"],
                    help="Keep specific pipeline in VRAM (only with --pipeline-swap)")

args = parser.parse_args()

# Create directories
os.makedirs("generated_images", exist_ok=True)
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("loras", exist_ok=True)

# LoRA paths
LORAS_DIR = "loras"
PRESETS_FILE = os.path.join(LORAS_DIR, "presets.json")

# Enums
class JobType(str, Enum):
    GENERATE = "generate"
    EDIT = "edit"

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 512
    height: Optional[int] = 512
    num_inference_steps: Optional[int] = 20
    cfg_scale: Optional[float] = 4.0
    seed: Optional[int] = None

class EditRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = " "
    num_inference_steps: Optional[int] = 50
    cfg_scale: Optional[float] = 4.0
    seed: Optional[int] = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    job_type: JobType
    status: str  # "queued", "processing", "completed", "failed"
    progress: Optional[float] = None
    input_image_url: Optional[str] = None
    output_image_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    cfg_scale: Optional[float] = None

class LoRASource(str, Enum):
    huggingface = "huggingface"
    local = "local"

class LoRARequest(BaseModel):
    name: str  # Display name for the LoRA
    source: LoRASource = LoRASource.huggingface  # Source: huggingface or local
    repo_id: Optional[str] = None  # HuggingFace repo ID (required for HuggingFace source)
    weight_name: Optional[str] = None  # Filename for local, optional for HF
    scale: Optional[float] = 1.0  # LoRA scale/strength
    pipeline: Literal["generation", "edit", "both"] = "edit"  # Which pipeline to apply to

class LoRAPreset(BaseModel):
    name: str
    source: LoRASource
    repo_id: Optional[str] = None
    weight_name: Optional[str] = None
    pipeline: str
    description: str
    recommended_steps: Optional[int] = None

class LoRAInfo(BaseModel):
    name: str
    source: LoRASource
    repo_id: Optional[str] = None
    weight_name: Optional[str] = None
    scale: float
    pipeline: str
    loaded_at: str
    active: bool

# Global state
generation_pipeline = None
edit_pipeline = None
job_queue = Queue()
job_status: Dict[str, JobStatus] = {}
current_job_id = None
loaded_loras: Dict[str, Dict[str, Any]] = {}  # Track loaded LoRAs
pipeline_lock = threading.Lock()  # For thread-safe pipeline swapping
current_pipeline_in_vram = None  # Track which pipeline is currently in VRAM

# Determine device
def get_device():
    if args.device == "cpu":
        return "cpu", torch.float32
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available, falling back to CPU")
            return "cpu", torch.float32
        return "cuda", torch.bfloat16
    else:  # auto
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        return "cpu", torch.float32

def swap_pipeline_to_device(pipeline_name: str):
    """Swap pipeline to GPU/CPU based on configuration"""
    global generation_pipeline, edit_pipeline, current_pipeline_in_vram
    
    if not args.pipeline_swap:
        return  # No swapping needed
    
    device, _ = get_device()
    if device == "cpu":
        return  # No swapping on CPU-only systems
    
    with pipeline_lock:
        # Determine which pipeline to swap in
        target_pipeline = generation_pipeline if pipeline_name == "generation" else edit_pipeline
        
        if current_pipeline_in_vram == pipeline_name:
            return  # Already in VRAM
        
        print(f"🔄 Swapping {pipeline_name} pipeline to VRAM...")
        
        # Move current pipeline out of VRAM (unless it's pinned)
        if current_pipeline_in_vram and current_pipeline_in_vram != args.keep_in_vram:
            if current_pipeline_in_vram == "generation" and generation_pipeline:
                print(f"📤 Moving generation pipeline to CPU...")
                # Disable CPU offload before moving to CPU
                if hasattr(generation_pipeline, 'disable_model_cpu_offload'):
                    generation_pipeline.disable_model_cpu_offload()
                generation_pipeline.to("cpu")
            elif current_pipeline_in_vram == "edit" and edit_pipeline:
                print(f"📤 Moving edit pipeline to CPU...")
                # Disable CPU offload before moving to CPU
                if hasattr(edit_pipeline, 'disable_model_cpu_offload'):
                    edit_pipeline.disable_model_cpu_offload()
                edit_pipeline.to("cpu")
            torch.cuda.empty_cache()
        
        # Move target pipeline to VRAM
        if target_pipeline:
            print(f"📥 Moving {pipeline_name} pipeline to VRAM...")
            target_pipeline.to("cuda")
            
            # Re-enable CPU offload if configured
            if args.cpu_offload:
                print(f"   Enabling CPU offload for {pipeline_name} pipeline...")
                target_pipeline.enable_model_cpu_offload()
            
            current_pipeline_in_vram = pipeline_name
        
        torch.cuda.empty_cache()
        print(f"✅ Pipeline swap complete")

def resize_image_if_needed(image, max_pixels=None):
    """Resize image if it exceeds max pixels while maintaining aspect ratio."""
    if max_pixels is None:
        max_pixels = args.max_pixels
        
    width, height = image.size
    current_pixels = width * height
    
    if current_pixels <= max_pixels:
        return image
    
    scale_factor = (max_pixels / current_pixels) ** 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Ensure dimensions are multiples of 4
    new_width = new_width - (new_width % 4)
    new_height = new_height - (new_height % 4)
    
    # Ensure minimum dimensions
    new_width = max(new_width, 256)
    new_height = max(new_height, 256)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

def load_generation_pipeline():
    """Load the Qwen-Image generation pipeline"""
    global generation_pipeline, current_pipeline_in_vram

    if args.disable_generation:
        print("⚠️ Image generation pipeline disabled")
        return

    # Select model based on argument
    if args.generation_model == "2512":
        model_id = "Qwen/Qwen-Image-2512"
        model_label = "Qwen-Image-2512"
    else:
        model_id = "Qwen/Qwen-Image"
        model_label = "Qwen-Image (original)"

    print(f"🔄 Loading {model_label} generation pipeline...")

    device, torch_dtype = get_device()

    # If pipeline swapping is enabled, always load to CPU first
    load_device = "cpu" if args.pipeline_swap else device

    generation_pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    # CPU offloading works differently with pipeline swap
    if args.cpu_offload and device == "cuda":
        if args.pipeline_swap:
            # We'll enable CPU offload after swapping to GPU
            generation_pipeline = generation_pipeline.to(load_device)
        else:
            # Normal CPU offload when not swapping
            generation_pipeline.enable_model_cpu_offload()
    else:
        generation_pipeline = generation_pipeline.to(load_device)

    # If this pipeline should be kept in VRAM, move it now
    if args.pipeline_swap and args.keep_in_vram == "generation" and device == "cuda":
        generation_pipeline.to("cuda")
        if args.cpu_offload:
            generation_pipeline.enable_model_cpu_offload()
        current_pipeline_in_vram = "generation"

    print(f"✅ {model_label} generation pipeline loaded successfully!")
    if args.pipeline_swap:
        print(f"   Pipeline swap mode enabled - will swap to VRAM on demand")
        if args.keep_in_vram == "generation":
            print(f"   Generation pipeline pinned in VRAM")
    if args.cpu_offload:
        print(f"   CPU offloading enabled for layer-by-layer processing")

def load_edit_pipeline_original():
    """Load the original Qwen-Image-Edit pipeline with optional quantization"""
    global edit_pipeline, current_pipeline_in_vram
    from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
    
    print("🔄 Loading original Qwen-Image-Edit pipeline...")
    
    model_id = "Qwen/Qwen-Image-Edit"
    device, torch_dtype = get_device()
    
    # If pipeline swapping is enabled, always load to CPU first
    load_device = "cpu" if args.pipeline_swap else device
    
    if args.quantize and device == "cuda":
        print("📦 Using 4-bit quantization...")
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        from transformers import Qwen2_5_VLForConditionalGeneration
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
        
        # Load Visual Transformer (4-bit)
        print("1/4 - Loading visual transformer...")
        quantization_config_diffusers = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config_diffusers,
            torch_dtype=torch_dtype,
        )
        
        # Load Text Encoder (4-bit)
        print("2/4 - Loading text encoder...")
        quantization_config_transformers = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            quantization_config=quantization_config_transformers,
            torch_dtype=torch_dtype,
        )
        
        # Create Pipeline
        print("3/4 - Creating pipeline...")
        edit_pipeline = QwenImageEditPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch_dtype,
        )
    else:
        # Load without quantization
        edit_pipeline = QwenImageEditPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype
        )
    
    # Apply CPU offloading or move to device
    if args.cpu_offload and device == "cuda":
        if args.pipeline_swap:
            # We'll enable CPU offload after swapping to GPU
            edit_pipeline = edit_pipeline.to(load_device)
        else:
            # Normal CPU offload when not swapping
            edit_pipeline.enable_model_cpu_offload()
    else:
        edit_pipeline = edit_pipeline.to(load_device)
    
    # If this pipeline should be kept in VRAM, move it now
    if args.pipeline_swap and args.keep_in_vram == "edit" and device == "cuda":
        edit_pipeline.to("cuda")
        if args.cpu_offload:
            edit_pipeline.enable_model_cpu_offload()
        current_pipeline_in_vram = "edit"
    
    edit_pipeline.set_progress_bar_config(disable=None)
    print(f"✅ Original edit pipeline loaded successfully!")
    if args.pipeline_swap:
        print(f"   Pipeline swap mode enabled - will swap to VRAM on demand")
        if args.keep_in_vram == "edit":
            print(f"   Edit pipeline pinned in VRAM")
    if args.cpu_offload:
        print(f"   CPU offloading enabled for layer-by-layer processing")

def load_edit_pipeline_plus(model_version: str):
    """Load the Qwen-Image-Edit-2509 or 2511 pipeline (QwenImageEditPlusPipeline)"""
    global edit_pipeline, current_pipeline_in_vram
    from diffusers import QwenImageEditPlusPipeline

    model_id = f"Qwen/Qwen-Image-Edit-{model_version}"
    print(f"🔄 Loading Qwen-Image-Edit-{model_version} pipeline...")

    device, torch_dtype = get_device()

    # If pipeline swapping is enabled, always load to CPU first
    load_device = "cpu" if args.pipeline_swap else device

    edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype
    )

    # Move to device with CPU offload support
    if args.cpu_offload and device == "cuda":
        if args.pipeline_swap:
            # We'll enable CPU offload after swapping to GPU
            edit_pipeline.to(load_device)
        else:
            # Try to enable CPU offload
            if hasattr(edit_pipeline, 'enable_model_cpu_offload'):
                edit_pipeline.enable_model_cpu_offload()
            else:
                # Fallback to regular device placement
                edit_pipeline.to(device)
                print("⚠️ CPU offload not supported for QwenImageEditPlusPipeline")
    else:
        edit_pipeline.to(load_device)

    # If this pipeline should be kept in VRAM, move it now
    if args.pipeline_swap and args.keep_in_vram == "edit" and device == "cuda":
        edit_pipeline.to("cuda")
        if args.cpu_offload and hasattr(edit_pipeline, 'enable_model_cpu_offload'):
            edit_pipeline.enable_model_cpu_offload()
        current_pipeline_in_vram = "edit"

    edit_pipeline.set_progress_bar_config(disable=None)

    print(f"✅ Qwen-Image-Edit-{model_version} pipeline loaded successfully!")
    if model_version == "2511":
        print("Features: Reduced image drift, improved consistency, enhanced geometric reasoning")
    else:
        print("Features: Enhanced consistency, multi-image support, native ControlNet")
    if args.pipeline_swap:
        print(f"   Pipeline swap mode enabled - will swap to VRAM on demand")
        if args.keep_in_vram == "edit":
            print(f"   Edit pipeline pinned in VRAM")
    if args.cpu_offload:
        print(f"   CPU offloading enabled (if supported)")

def load_edit_pipeline():
    """Load the appropriate edit pipeline based on args"""
    global edit_pipeline

    if args.disable_edit:
        print("⚠️ Image editing pipeline disabled")
        return

    if args.edit_model in ["2509", "2511"]:
        load_edit_pipeline_plus(args.edit_model)
    else:
        load_edit_pipeline_original()

def process_job_queue():
    """Background worker to process both generation and edit requests"""
    global current_job_id
    
    device, _ = get_device()
    
    while True:
        try:
            # Get next job from queue
            job_data = job_queue.get(timeout=1)
            job_id = job_data["job_id"]
            job_type = job_data["job_type"]
            request = job_data["request"]
            
            current_job_id = job_id
            
            # Update status to processing
            job_status[job_id].status = "processing"
            job_status[job_id].progress = 0.1
            
            print(f"Processing {job_type} job {job_id}: {request.prompt[:50]}...")
            
            # Swap pipeline to VRAM if needed
            if args.pipeline_swap:
                swap_pipeline_to_device("generation" if job_type == JobType.GENERATE else "edit")
            
            # Set seed
            if request.seed is None:
                generator = torch.Generator(device=device if device == "cuda" else "cpu")
            else:
                generator = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(request.seed)
            
            if job_type == JobType.GENERATE:
                # Generation job
                job_status[job_id].progress = 0.3
                
                image = generation_pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.num_inference_steps,
                    true_cfg_scale=request.cfg_scale,
                    generator=generator
                ).images[0]
                
            else:  # Edit job
                input_image_path = job_data["input_image_path"]
                
                # Load and resize input image
                image = Image.open(input_image_path).convert("RGB")
                image = resize_image_if_needed(image)
                job_status[job_id].progress = 0.2
                
                # Prepare inputs based on pipeline type
                if args.edit_model == "2509":
                    inputs = {
                        "image": image,
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "num_inference_steps": request.num_inference_steps,
                        "true_cfg_scale": request.cfg_scale,
                        "generator": generator,
                        "guidance_scale": 1.0,
                        "num_images_per_prompt": 1
                    }
                else:
                    inputs = {
                        "image": image,
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "num_inference_steps": request.num_inference_steps,
                        "true_cfg_scale": request.cfg_scale,
                        "generator": generator
                    }
                
                job_status[job_id].progress = 0.3
                
                # Generate edited image
                with torch.inference_mode():
                    output = edit_pipeline(**inputs)
                    image = output.images[0]
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save image
            job_status[job_id].progress = 0.9
            output_filename = f"{job_id}_output.png"
            output_path = os.path.join("generated_images", output_filename)
            image.save(output_path)
            
            # Update job status
            job_status[job_id].status = "completed"
            job_status[job_id].progress = 1.0
            job_status[job_id].output_image_url = f"/images/{output_filename}"
            job_status[job_id].completed_at = datetime.now().isoformat()
            
            print(f"Job {job_id} completed successfully!")
            
            current_job_id = None
            
        except torch.cuda.OutOfMemoryError:
            if current_job_id:
                job_status[current_job_id].status = "failed"
                job_status[current_job_id].error = "Out of GPU memory. Try enabling --pipeline-swap or --quantize"
                print(f"Job {current_job_id} failed: Out of GPU memory")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            current_job_id = None
            
        except Exception as e:
            if current_job_id:
                job_status[current_job_id].status = "failed"
                job_status[current_job_id].error = str(e)
                print(f"Job {current_job_id} failed: {e}")
            
            current_job_id = None
            time.sleep(1)

# Start background worker thread
worker_thread = threading.Thread(target=process_job_queue, daemon=True)
worker_thread.start()

@app.on_event("startup")
async def startup_event():
    """Load pipelines when server starts"""
    load_generation_pipeline()
    load_edit_pipeline()

@app.post("/generate", response_model=JobResponse)
async def generate_image(request: GenerationRequest):
    """Submit a new image generation request"""
    if generation_pipeline is None:
        raise HTTPException(status_code=503, detail="Generation pipeline is disabled")
        
    job_id = str(uuid.uuid4())
    
    # Create job status
    job_status[job_id] = JobStatus(
        job_id=job_id,
        job_type=JobType.GENERATE,
        status="queued",
        created_at=datetime.now().isoformat(),
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        cfg_scale=request.cfg_scale
    )
    
    # Add to queue
    job_queue.put({
        "job_id": job_id,
        "job_type": JobType.GENERATE,
        "request": request
    })
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Generation job submitted. Queue position: {job_queue.qsize()}"
    )

@app.post("/edit", response_model=JobResponse)
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(" "),
    num_inference_steps: int = Form(50),
    cfg_scale: float = Form(4.0),
    seed: Optional[int] = Form(None)
):
    """Submit a new image edit request"""
    if edit_pipeline is None:
        raise HTTPException(status_code=503, detail="Edit pipeline is disabled")
        
    job_id = str(uuid.uuid4())
    
    # Save uploaded image
    input_filename = f"{job_id}_input.png"
    input_path = os.path.join("uploaded_images", input_filename)
    
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img.save(input_path)
    
    # Create request object
    request = EditRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        seed=seed
    )
    
    # Create job status
    job_status[job_id] = JobStatus(
        job_id=job_id,
        job_type=JobType.EDIT,
        status="queued",
        created_at=datetime.now().isoformat(),
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        input_image_url=f"/uploads/{input_filename}"
    )
    
    # Add to queue
    job_queue.put({
        "job_id": job_id,
        "job_type": JobType.EDIT,
        "request": request,
        "input_image_path": input_path
    })
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Edit job submitted. Queue position: {job_queue.qsize()}"
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.get("/queue")
async def get_queue_info():
    """Get information about the current queue"""
    generation_jobs = len([j for j in job_status.values() if j.job_type == JobType.GENERATE])
    edit_jobs = len([j for j in job_status.values() if j.job_type == JobType.EDIT])
    
    return {
        "queue_size": job_queue.qsize(),
        "current_job": current_job_id,
        "total_jobs": len(job_status),
        "generation_jobs": generation_jobs,
        "edit_jobs": edit_jobs,
        "completed_jobs": len([j for j in job_status.values() if j.status == "completed"]),
        "failed_jobs": len([j for j in job_status.values() if j.status == "failed"])
    }

@app.get("/jobs")
async def get_all_jobs(job_type: Optional[JobType] = None):
    """Get all job statuses, optionally filtered by type"""
    jobs = list(job_status.values())
    
    if job_type:
        jobs = [j for j in jobs if j.job_type == job_type]
    
    return jobs

@app.get("/system/info")
async def get_system_info():
    """Get system information including GPU memory usage"""
    device, dtype = get_device()
    
    info = {
        "device": device,
        "dtype": str(dtype),
        "cuda_available": torch.cuda.is_available(),
        "generation_pipeline": f"loaded ({args.generation_model})" if generation_pipeline else "disabled",
        "generation_model": args.generation_model,
        "edit_pipeline": f"loaded ({args.edit_model})" if edit_pipeline else "disabled",
        "edit_model": args.edit_model,
        "quantization": args.quantize,
        "cpu_offload": args.cpu_offload,
        "pipeline_swap": args.pipeline_swap,
        "keep_in_vram": args.keep_in_vram,
        "current_pipeline_in_vram": current_pipeline_in_vram,
        "max_edit_pixels": args.max_pixels,
        "server_args": vars(args)
    }

    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        })

    return info

@app.post("/lora/load")
async def load_lora(request: LoRARequest):
    """Load a LoRA adapter from HuggingFace or local file"""
    try:
        # Validate pipeline availability
        if request.pipeline in ["generation", "both"] and generation_pipeline is None:
            raise HTTPException(status_code=400, detail="Generation pipeline is not loaded")
        if request.pipeline in ["edit", "both"] and edit_pipeline is None:
            raise HTTPException(status_code=400, detail="Edit pipeline is not loaded")

        # Validate request based on source
        if request.source == LoRASource.huggingface:
            if not request.repo_id:
                raise HTTPException(status_code=400, detail="repo_id is required for HuggingFace source")
        elif request.source == LoRASource.local:
            if not request.weight_name:
                raise HTTPException(status_code=400, detail="weight_name is required for local source")
            local_path = os.path.join(LORAS_DIR, request.weight_name)
            if not os.path.exists(local_path):
                raise HTTPException(status_code=404, detail=f"Local LoRA file not found: {request.weight_name}")

        # Load LoRA on the appropriate pipeline(s)
        loaded_on = []

        def load_on_pipeline(pipeline, pipeline_name):
            if request.source == LoRASource.local:
                # Load from local file
                local_path = os.path.join(LORAS_DIR, request.weight_name)
                pipeline.load_lora_weights(
                    LORAS_DIR,
                    weight_name=request.weight_name,
                    adapter_name=request.name
                )
            else:
                # Load from HuggingFace
                if request.weight_name:
                    pipeline.load_lora_weights(
                        request.repo_id,
                        weight_name=request.weight_name,
                        adapter_name=request.name
                    )
                else:
                    pipeline.load_lora_weights(
                        request.repo_id,
                        adapter_name=request.name
                    )
            pipeline.set_adapters([request.name], adapter_weights=[request.scale])
            loaded_on.append(pipeline_name)

        if request.pipeline in ["generation", "both"] and generation_pipeline:
            load_on_pipeline(generation_pipeline, "generation")

        if request.pipeline in ["edit", "both"] and edit_pipeline:
            load_on_pipeline(edit_pipeline, "edit")

        # Track loaded LoRA
        loaded_loras[request.name] = {
            "info": LoRAInfo(
                name=request.name,
                source=request.source,
                repo_id=request.repo_id,
                weight_name=request.weight_name,
                scale=request.scale,
                pipeline=request.pipeline,
                loaded_at=datetime.now().isoformat(),
                active=True
            ),
            "loaded_on": loaded_on
        }

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "message": f"LoRA '{request.name}' loaded successfully",
            "source": request.source.value,
            "loaded_on": loaded_on
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LoRA: {str(e)}")

@app.post("/lora/unload/{name}")
async def unload_lora(name: str):
    """Unload a specific LoRA adapter"""
    if name not in loaded_loras:
        raise HTTPException(status_code=404, detail=f"LoRA '{name}' not found")
    
    try:
        lora_info = loaded_loras[name]
        unloaded_from = []
        
        # Unload from pipelines
        if "generation" in lora_info["loaded_on"] and generation_pipeline:
            generation_pipeline.unload_lora_weights()
            unloaded_from.append("generation")
        
        if "edit" in lora_info["loaded_on"] and edit_pipeline:
            edit_pipeline.unload_lora_weights()
            unloaded_from.append("edit")
        
        # Remove from tracking
        del loaded_loras[name]
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "message": f"LoRA '{name}' unloaded successfully",
            "unloaded_from": unloaded_from
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload LoRA: {str(e)}")

@app.post("/lora/activate/{name}")
async def activate_lora(name: str, scale: Optional[float] = None):
    """Activate a loaded LoRA with optional scale adjustment"""
    if name not in loaded_loras:
        raise HTTPException(status_code=404, detail=f"LoRA '{name}' not found")

    try:
        lora_info = loaded_loras[name]
        if scale is not None:
            lora_info["info"].scale = scale

        activated_on = []

        # Use enable_lora() to reactivate, then set_adapters() within no_grad for scale
        with torch.no_grad():
            if "generation" in lora_info["loaded_on"] and generation_pipeline:
                generation_pipeline.enable_lora()
                generation_pipeline.set_adapters([name], adapter_weights=[lora_info["info"].scale])
                activated_on.append("generation")

            if "edit" in lora_info["loaded_on"] and edit_pipeline:
                edit_pipeline.enable_lora()
                edit_pipeline.set_adapters([name], adapter_weights=[lora_info["info"].scale])
                activated_on.append("edit")

        lora_info["info"].active = True

        return {
            "message": f"LoRA '{name}' activated with scale {lora_info['info'].scale}",
            "activated_on": activated_on
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate LoRA: {str(e)}")

@app.post("/lora/deactivate/{name}")
async def deactivate_lora(name: str):
    """Deactivate a LoRA without unloading it"""
    if name not in loaded_loras:
        raise HTTPException(status_code=404, detail=f"LoRA '{name}' not found")
    
    try:
        lora_info = loaded_loras[name]
        deactivated_on = []
        
        if "generation" in lora_info["loaded_on"] and generation_pipeline:
            generation_pipeline.disable_lora()
            deactivated_on.append("generation")
        
        if "edit" in lora_info["loaded_on"] and edit_pipeline:
            edit_pipeline.disable_lora()
            deactivated_on.append("edit")
        
        lora_info["info"].active = False
        
        return {
            "message": f"LoRA '{name}' deactivated",
            "deactivated_on": deactivated_on
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deactivate LoRA: {str(e)}")

@app.get("/lora/list")
async def list_loras():
    """List all loaded LoRAs"""
    return {
        "loras": [lora["info"] for lora in loaded_loras.values()],
        "count": len(loaded_loras)
    }

@app.get("/lora/available")
async def get_available_loras():
    """List available local LoRA files from the loras/ directory"""
    try:
        available = []
        # Find all .safetensors files in the loras directory
        pattern = os.path.join(LORAS_DIR, "*.safetensors")
        files = globmodule.glob(pattern)

        for filepath in files:
            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath)
            modified = datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()

            available.append({
                "filename": filename,
                "size_mb": round(filesize / (1024 * 1024), 2),
                "modified": modified
            })

        # Sort by modification time (newest first)
        available.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "available": available,
            "count": len(available),
            "directory": LORAS_DIR
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan loras directory: {str(e)}")

@app.get("/lora/presets")
async def get_lora_presets():
    """Get configured LoRA presets from presets.json"""
    try:
        if not os.path.exists(PRESETS_FILE):
            return {"presets": [], "count": 0}

        with open(PRESETS_FILE, "r") as f:
            data = json.load(f)

        presets = data.get("presets", [])

        # Validate and convert to LoRAPreset models
        validated_presets = []
        for preset in presets:
            try:
                validated = LoRAPreset(
                    name=preset["name"],
                    source=LoRASource(preset.get("source", "huggingface")),
                    repo_id=preset.get("repo_id"),
                    weight_name=preset.get("weight_name"),
                    pipeline=preset.get("pipeline", "edit"),
                    description=preset.get("description", ""),
                    recommended_steps=preset.get("recommended_steps")
                )
                validated_presets.append(validated)
            except Exception as e:
                print(f"Warning: Invalid preset skipped: {e}")

        return {
            "presets": validated_presets,
            "count": len(validated_presets)
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid presets.json format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load presets: {str(e)}")

# Serve generated images
app.mount("/images", StaticFiles(directory="generated_images"), name="images")

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory="uploaded_images"), name="uploads")

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    print("Starting Qwen Image Studio Server...")
    print(f"Configuration: {vars(args)}")
    uvicorn.run(app, host=args.host, port=args.port)