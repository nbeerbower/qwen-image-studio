from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal, List
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
parser.add_argument("--edit-model", type=str, default="2511",
                    help="Edit model: 'original', '2509', '2511', or path to local model/checkpoint")
parser.add_argument("--edit-model-type", type=str, default=None, choices=["original", "plus"],
                    help="Pipeline type for local models: 'original' or 'plus' (auto-detected if not set)")
parser.add_argument("--generation-model", type=str, default="2512", choices=["original", "2512"],
                    help="Which generation model to use: 'original' or '2512' (default: 2512)")
parser.add_argument("--quantize", action="store_true",
                    help="Enable 4-bit quantization for reduced VRAM usage (incompatible with --pipeline-swap)")
parser.add_argument("--cpu-offload", action="store_true", default=True,
                    help="Enable CPU offloading (default: True)")
parser.add_argument("--max-pixels", type=int, default=1048576,
                    help="Maximum pixels for image editing (default: 1048576)")
parser.add_argument("--disable-generation", action="store_true",
                    help="Disable image generation pipeline")
parser.add_argument("--disable-edit", action="store_true",
                    help="Disable image editing pipeline")
parser.add_argument("--device", type=str, default="auto",
                    help="Device to use: auto, cpu, cuda, cuda:0, cuda:1, etc. (default: auto)")
parser.add_argument("--generation-device", type=str, default=None,
                    help="Device for generation pipeline (overrides --device). E.g., cuda:0")
parser.add_argument("--edit-device", type=str, default=None,
                    help="Device for edit pipeline (overrides --device). E.g., cuda:1")
parser.add_argument("--device-map", action="store_true",
                    help="Use accelerate device_map='auto' to distribute model across GPUs")
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
    input_image_url: Optional[str] = None  # Deprecated, use input_image_urls
    input_image_urls: Optional[List[str]] = None  # Support multiple input images
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
def parse_device(device_str):
    """Parse device string and return (device, dtype)"""
    if device_str == "cpu":
        return "cpu", torch.float32
    elif device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"⚠️ {device_str} requested but CUDA not available, falling back to CPU")
            return "cpu", torch.float32
        # Handle cuda:N format
        if ":" in device_str:
            gpu_id = int(device_str.split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                print(f"⚠️ {device_str} requested but only {torch.cuda.device_count()} GPUs available, using cuda:0")
                return "cuda:0", torch.bfloat16
        return device_str, torch.bfloat16
    else:  # auto
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        return "cpu", torch.float32

def get_device():
    """Get default device"""
    return parse_device(args.device)

def get_generation_device():
    """Get device for generation pipeline"""
    if args.generation_device:
        return parse_device(args.generation_device)
    return get_device()

def get_edit_device():
    """Get device for edit pipeline"""
    if args.edit_device:
        return parse_device(args.edit_device)
    return get_device()

def get_gpu_count():
    """Get number of available GPUs"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

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
    from diffusers import QwenImageTransformer2DModel

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

    device, torch_dtype = get_generation_device()
    if args.generation_device:
        print(f"   Using device: {device}")

    # If pipeline swapping is enabled, always load to CPU first
    load_device = "cpu" if args.pipeline_swap else device

    # Device map for multi-GPU model parallelism
    if args.device_map and get_gpu_count() > 1:
        print(f"📊 Using device_map='balanced' to distribute across {get_gpu_count()} GPUs")
        generation_pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="balanced",
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        print(f"✅ {model_label} generation pipeline loaded with multi-GPU distribution!")
        return

    if args.quantize and device != "cpu":
        # Quantized models cannot be moved between devices
        if args.pipeline_swap:
            print("⚠️ Warning: --quantize is incompatible with --pipeline-swap. Disabling pipeline swap.")
            args.pipeline_swap = False

        print("📦 Using 4-bit quantization...")
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        from transformers import Qwen2_5_VLForConditionalGeneration
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

        # Load Visual Transformer (4-bit)
        print("1/3 - Loading visual transformer...")
        quantization_config_diffusers = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config_diffusers,
            torch_dtype=torch_dtype,
            device_map={"": device},
        )

        # Load Text Encoder (4-bit)
        print("2/3 - Loading text encoder...")
        quantization_config_transformers = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            quantization_config=quantization_config_transformers,
            torch_dtype=torch_dtype,
            device_map={"": device},
        )

        # Create Pipeline
        print("3/3 - Creating pipeline...")
        generation_pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        # Move VAE to the same device (it's not quantized)
        generation_pipeline.vae.to(device)
        print(f"✅ {model_label} generation pipeline loaded with 4-bit quantization on {device}!")
        return

    # Non-quantized loading
    generation_pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    is_cuda = device.startswith("cuda")

    # CPU offloading works differently with pipeline swap
    if args.cpu_offload and is_cuda:
        if args.pipeline_swap:
            # We'll enable CPU offload after swapping to GPU
            generation_pipeline = generation_pipeline.to(load_device)
        else:
            # Normal CPU offload when not swapping
            generation_pipeline.enable_model_cpu_offload()
    else:
        generation_pipeline = generation_pipeline.to(load_device)

    # If this pipeline should be kept in VRAM, move it now
    if args.pipeline_swap and args.keep_in_vram == "generation" and is_cuda:
        generation_pipeline.to(device)
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
    device, torch_dtype = get_edit_device()
    if args.edit_device:
        print(f"   Using device: {device}")

    is_cuda = device.startswith("cuda")

    # If pipeline swapping is enabled, always load to CPU first
    load_device = "cpu" if args.pipeline_swap else device

    # Device map for multi-GPU model parallelism
    if args.device_map and get_gpu_count() > 1:
        print(f"📊 Using device_map='balanced' to distribute across {get_gpu_count()} GPUs")
        edit_pipeline = QwenImageEditPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="balanced",
        )
        edit_pipeline.set_progress_bar_config(disable=None)
        print("✅ Original edit pipeline loaded with multi-GPU distribution!")
        return

    if args.quantize and is_cuda:
        # Quantized models cannot be moved between devices, so check for conflicts
        if args.pipeline_swap:
            print("⚠️ Warning: --quantize is incompatible with --pipeline-swap. Disabling pipeline swap.")
            args.pipeline_swap = False

        print("📦 Using 4-bit quantization...")
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        from transformers import Qwen2_5_VLForConditionalGeneration
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

        # Load Visual Transformer (4-bit)
        print("1/3 - Loading visual transformer...")
        quantization_config_diffusers = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config_diffusers,
            torch_dtype=torch_dtype,
            device_map={"": device},
        )

        # Load Text Encoder (4-bit)
        print("2/3 - Loading text encoder...")
        quantization_config_transformers = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            quantization_config=quantization_config_transformers,
            torch_dtype=torch_dtype,
            device_map={"": device},
        )

        # Create Pipeline
        print("3/3 - Creating pipeline...")
        edit_pipeline = QwenImageEditPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch_dtype,
        )
        # Move VAE to the same device (it's not quantized)
        edit_pipeline.vae.to(device)
        print(f"✅ Original edit pipeline loaded with 4-bit quantization on {device}!")
        edit_pipeline.set_progress_bar_config(disable=None)
        return
    else:
        # Load without quantization
        edit_pipeline = QwenImageEditPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype
        )

    # Apply CPU offloading or move to device
    if args.cpu_offload and is_cuda:
        if args.pipeline_swap:
            # We'll enable CPU offload after swapping to GPU
            edit_pipeline = edit_pipeline.to(load_device)
        else:
            # Normal CPU offload when not swapping
            edit_pipeline.enable_model_cpu_offload()
    else:
        edit_pipeline = edit_pipeline.to(load_device)

    # If this pipeline should be kept in VRAM, move it now
    if args.pipeline_swap and args.keep_in_vram == "edit" and is_cuda:
        edit_pipeline.to(device)
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
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

    model_id = f"Qwen/Qwen-Image-Edit-{model_version}"
    print(f"🔄 Loading Qwen-Image-Edit-{model_version} pipeline...")

    device, torch_dtype = get_edit_device()
    if args.edit_device:
        print(f"   Using device: {device}")

    is_cuda = device.startswith("cuda")

    # If pipeline swapping is enabled, always load to CPU first
    load_device = "cpu" if args.pipeline_swap else device

    # Device map for multi-GPU model parallelism
    if args.device_map and get_gpu_count() > 1:
        print(f"📊 Using device_map='balanced' to distribute across {get_gpu_count()} GPUs")
        edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="balanced",
        )
        edit_pipeline.set_progress_bar_config(disable=None)
        print(f"✅ Qwen-Image-Edit-{model_version} pipeline loaded with multi-GPU distribution!")
        if model_version == "2511":
            print("Features: Reduced image drift, improved consistency, enhanced geometric reasoning")
        else:
            print("Features: Enhanced consistency, multi-image support, native ControlNet")
        return

    if args.quantize and is_cuda:
        # Quantized models cannot be moved between devices, so check for conflicts
        if args.pipeline_swap:
            print("⚠️ Warning: --quantize is incompatible with --pipeline-swap. Disabling pipeline swap.")
            args.pipeline_swap = False

        print("📦 Using 4-bit quantization...")
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        from transformers import Qwen2_5_VLForConditionalGeneration
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

        # Load Visual Transformer (4-bit)
        print("1/3 - Loading visual transformer...")
        quantization_config_diffusers = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config_diffusers,
            torch_dtype=torch_dtype,
            device_map={"": device},
        )

        # Load Text Encoder (4-bit)
        print("2/3 - Loading text encoder...")
        quantization_config_transformers = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            quantization_config=quantization_config_transformers,
            torch_dtype=torch_dtype,
            device_map={"": device},
        )

        # Create Pipeline
        print("3/3 - Creating pipeline...")
        edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch_dtype,
        )
        # Move VAE to the same device (it's not quantized)
        edit_pipeline.vae.to(device)
        edit_pipeline.set_progress_bar_config(disable=None)
        print(f"✅ Qwen-Image-Edit-{model_version} pipeline loaded with 4-bit quantization on {device}!")
        if model_version == "2511":
            print("Features: Reduced image drift, improved consistency, enhanced geometric reasoning")
        else:
            print("Features: Enhanced consistency, multi-image support, native ControlNet")
        return

    # Non-quantized loading
    edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype
    )

    # Move to device with CPU offload support
    if args.cpu_offload and is_cuda:
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
    if args.pipeline_swap and args.keep_in_vram == "edit" and is_cuda:
        edit_pipeline.to(device)
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

def load_edit_pipeline_from_single_file(checkpoint_path: str):
    """Load edit pipeline from a single .safetensors checkpoint (ComfyUI format)"""
    global edit_pipeline, current_pipeline_in_vram
    from safetensors import safe_open
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, AutoencoderKL
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

    print(f"🔄 Loading edit pipeline from single file: {checkpoint_path}")

    device, torch_dtype = get_edit_device()
    if args.edit_device:
        print(f"   Using device: {device}")

    is_cuda = device.startswith("cuda")

    # Determine pipeline type
    pipeline_type = args.edit_model_type or "plus"
    print(f"   Pipeline type: {pipeline_type}")

    # Load the checkpoint
    print("1/4 - Loading checkpoint...")
    state_dict = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # Separate components
    print("2/4 - Separating components...")
    transformer_state = {}
    text_encoder_state = {}
    vae_state = {}

    for key, tensor in state_dict.items():
        if key.startswith("model.diffusion_model."):
            # Remove prefix for transformer
            new_key = key.replace("model.diffusion_model.", "")
            transformer_state[new_key] = tensor
        elif key.startswith("text_encoders."):
            # Remove prefix for text encoder (handle qwen25_7b or other variants)
            parts = key.split(".", 2)
            if len(parts) >= 3:
                new_key = parts[2]  # Skip "text_encoders.qwen25_7b."
                text_encoder_state[new_key] = tensor
        elif key.startswith("vae."):
            new_key = key.replace("vae.", "")
            vae_state[new_key] = tensor

    print(f"   Transformer keys: {len(transformer_state)}")
    print(f"   Text encoder keys: {len(text_encoder_state)}")
    print(f"   VAE keys: {len(vae_state)}")

    # Load base pipeline for config, then replace weights
    print("3/4 - Loading base pipeline structure...")
    base_model = "Qwen/Qwen-Image-Edit-2511"  # Use 2511 as base for structure

    if args.device_map and get_gpu_count() > 1:
        print(f"📊 Using device_map='balanced' for multi-GPU")
        edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map="balanced",
        )
    else:
        edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

    # Load custom weights
    print("4/4 - Loading custom weights...")

    # Load transformer weights
    missing, unexpected = edit_pipeline.transformer.load_state_dict(transformer_state, strict=False)
    if missing:
        print(f"   Transformer - missing keys: {len(missing)}")
    if unexpected:
        print(f"   Transformer - unexpected keys: {len(unexpected)}")

    # Load text encoder weights if present
    if text_encoder_state:
        missing, unexpected = edit_pipeline.text_encoder.load_state_dict(text_encoder_state, strict=False)
        if missing:
            print(f"   Text encoder - missing keys: {len(missing)}")
        if unexpected:
            print(f"   Text encoder - unexpected keys: {len(unexpected)}")

    # Load VAE weights if present
    if vae_state:
        missing, unexpected = edit_pipeline.vae.load_state_dict(vae_state, strict=False)
        if missing:
            print(f"   VAE - missing keys: {len(missing)}")
        if unexpected:
            print(f"   VAE - unexpected keys: {len(unexpected)}")

    # Move to device if not using device_map
    if not (args.device_map and get_gpu_count() > 1):
        if args.cpu_offload and is_cuda:
            if hasattr(edit_pipeline, 'enable_model_cpu_offload'):
                edit_pipeline.enable_model_cpu_offload()
            else:
                edit_pipeline.to(device)
        else:
            edit_pipeline.to(device)

    edit_pipeline.set_progress_bar_config(disable=None)

    # Clear memory
    del state_dict, transformer_state, text_encoder_state, vae_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"✅ Custom edit pipeline loaded from {os.path.basename(checkpoint_path)}")

def load_edit_pipeline_from_local(model_path: str):
    """Load edit pipeline from a local directory (diffusers format)"""
    global edit_pipeline, current_pipeline_in_vram
    from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline

    print(f"🔄 Loading edit pipeline from local path: {model_path}")

    device, torch_dtype = get_edit_device()
    is_cuda = device.startswith("cuda")

    # Determine pipeline type
    pipeline_type = args.edit_model_type or "plus"
    PipelineClass = QwenImageEditPlusPipeline if pipeline_type == "plus" else QwenImageEditPipeline

    print(f"   Pipeline type: {pipeline_type}")

    if args.device_map and get_gpu_count() > 1:
        print(f"📊 Using device_map='balanced' for multi-GPU")
        edit_pipeline = PipelineClass.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="balanced",
        )
    else:
        edit_pipeline = PipelineClass.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        if args.cpu_offload and is_cuda:
            if hasattr(edit_pipeline, 'enable_model_cpu_offload'):
                edit_pipeline.enable_model_cpu_offload()
            else:
                edit_pipeline.to(device)
        else:
            edit_pipeline.to(device)

    edit_pipeline.set_progress_bar_config(disable=None)
    print(f"✅ Edit pipeline loaded from {model_path}")

def load_edit_pipeline():
    """Load the appropriate edit pipeline based on args"""
    global edit_pipeline

    if args.disable_edit:
        print("⚠️ Image editing pipeline disabled")
        return

    # Check if edit_model is a path
    if os.path.exists(args.edit_model):
        if args.edit_model.endswith('.safetensors'):
            # Single file checkpoint
            load_edit_pipeline_from_single_file(args.edit_model)
        elif os.path.isdir(args.edit_model):
            # Local directory in diffusers format
            load_edit_pipeline_from_local(args.edit_model)
        else:
            print(f"⚠️ Unknown model format: {args.edit_model}")
            return
    elif args.edit_model in ["2509", "2511"]:
        load_edit_pipeline_plus(args.edit_model)
    elif args.edit_model == "original":
        load_edit_pipeline_original()
    else:
        # Try as HuggingFace model ID
        print(f"🔄 Loading edit model from HuggingFace: {args.edit_model}")
        load_edit_pipeline_from_local(args.edit_model)

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
                input_image_paths = job_data.get("input_image_paths", [job_data.get("input_image_path")])

                # Load and resize input images
                input_images = []
                for path in input_image_paths:
                    img = Image.open(path).convert("RGB")
                    img = resize_image_if_needed(img)
                    input_images.append(img)

                job_status[job_id].progress = 0.2

                # Prepare inputs based on pipeline type
                if args.edit_model in ["2509", "2511"]:
                    # QwenImageEditPlusPipeline supports multiple images
                    # Pass single image or list depending on count
                    image_input = input_images if len(input_images) > 1 else input_images[0]
                    inputs = {
                        "image": image_input,
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "num_inference_steps": request.num_inference_steps,
                        "true_cfg_scale": request.cfg_scale,
                        "generator": generator,
                        "guidance_scale": 1.0,
                        "num_images_per_prompt": 1
                    }
                else:
                    # Original pipeline only supports single image
                    inputs = {
                        "image": input_images[0],
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
    images: List[UploadFile] = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(" "),
    num_inference_steps: int = Form(50),
    cfg_scale: float = Form(4.0),
    seed: Optional[int] = Form(None)
):
    """Submit a new image edit request (supports multiple images for 2509/2511 models)"""
    if edit_pipeline is None:
        raise HTTPException(status_code=503, detail="Edit pipeline is disabled")

    # Check if multi-image is supported
    supports_multi_image = args.edit_model in ["2509", "2511"]
    if len(images) > 1 and not supports_multi_image:
        raise HTTPException(
            status_code=400,
            detail=f"Multiple images only supported with edit models 2509/2511. Current model: {args.edit_model}"
        )

    job_id = str(uuid.uuid4())

    # Save uploaded images
    input_paths = []
    input_urls = []
    for i, image in enumerate(images):
        input_filename = f"{job_id}_input_{i}.png"
        input_path = os.path.join("uploaded_images", input_filename)

        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.save(input_path)

        input_paths.append(input_path)
        input_urls.append(f"/uploads/{input_filename}")

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
        input_image_url=input_urls[0],  # Backwards compatibility
        input_image_urls=input_urls
    )

    # Add to queue
    job_queue.put({
        "job_id": job_id,
        "job_type": JobType.EDIT,
        "request": request,
        "input_image_paths": input_paths  # Now a list
    })

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Edit job submitted with {len(images)} image(s). Queue position: {job_queue.qsize()}"
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