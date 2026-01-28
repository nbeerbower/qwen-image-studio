# Qwen Image Studio

A web-based interface for Qwen image generation and editing models with LoRA support, multi-image compositing, and flexible pipeline management.

## Features

- **Image Generation** - Text-to-image generation with Qwen-Image models
- **Image Editing** - Instruction-based image editing with multiple model options
- **Multi-Image Compositing** - Combine multiple images into one scene (2509/2511 models)
- **LoRA Support** - Load LoRAs from HuggingFace or local files
- **Preset System** - Configurable LoRA presets for quick loading
- **Pipeline Swapping** - Memory-efficient GPU management for limited VRAM
- **Job Queue** - Background processing with progress tracking

## Supported Models

### Generation Models
| Model | Flag | Description |
|-------|------|-------------|
| Qwen-Image | `--generation-model original` | Original generation model |
| Qwen-Image-2512 | `--generation-model 2512` | Latest generation model (default) |

### Edit Models
| Model | Flag | Description |
|-------|------|-------------|
| Qwen-Image-Edit | `--edit-model original` | Original edit model, supports Lightning LoRA |
| Qwen-Image-Edit-2509 | `--edit-model 2509` | Enhanced consistency, multi-image support |
| Qwen-Image-Edit-2511 | `--edit-model 2511` | Reduced drift, better geometric reasoning (default) |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd qwen-image-studio

# Install dependencies
pip install torch torchvision
pip install diffusers transformers accelerate
pip install fastapi uvicorn python-multipart
pip install pillow

# Optional: For quantization support
pip install bitsandbytes
```

## Quick Start

```bash
# Start with defaults (2512 generation + 2511 edit)
python server.py

# Open http://localhost:8000 in your browser
```

## Command Line Options

```
python server.py [OPTIONS]

Model Selection:
  --generation-model {original,2512}  Generation model (default: 2512)
  --edit-model {original,2509,2511}   Edit model (default: 2511)

Memory Management:
  --quantize              Enable 4-bit quantization for reduced VRAM
  --cpu-offload           Enable CPU offloading (default: True)
  --pipeline-swap         Swap pipelines between CPU/GPU on demand
  --keep-in-vram {generation,edit}  Pin specific pipeline in VRAM

Multi-GPU:
  --device DEVICE         Default device: auto, cpu, cuda, cuda:0, cuda:1, etc.
  --generation-device DEVICE  Device for generation (e.g., cuda:0)
  --edit-device DEVICE    Device for edit (e.g., cuda:1)
  --device-map            Distribute model across all GPUs (model parallelism)

Pipeline Control:
  --disable-generation    Disable generation pipeline
  --disable-edit          Disable edit pipeline

Other:
  --host HOST             Host to bind to (default: 0.0.0.0)
  --port PORT             Port to bind to (default: 8000)
  --max-pixels N          Max pixels for editing (default: 1048576)
```

## Usage Examples

### Memory-Constrained Setup (12GB VRAM)
```bash
python server.py --pipeline-swap --keep-in-vram edit
```

### Multi-GPU: Separate Pipelines on Different GPUs
```bash
# Generation on GPU 0, Edit on GPU 1
python server.py --generation-device cuda:0 --edit-device cuda:1
```

### Multi-GPU: Model Parallelism (Split Large Model Across GPUs)
```bash
# Distribute each model across all available GPUs
python server.py --device-map
```

### Fast Editing with Lightning LoRA
```bash
python server.py --edit-model original
# Then load Lightning LoRA from the UI for 8-step inference
```

### CPU-Only Mode
```bash
python server.py --device cpu --disable-generation
```

## Multi-Image Editing

The 2509 and 2511 edit models support combining multiple images:

1. Switch to Edit mode in the UI
2. Upload multiple images (drag & drop or click to add)
3. Write a prompt describing how to combine them:
   > "The cat from the first image is sitting next to the dog from the second image in a sunny garden"
4. Submit and wait for processing

## Multi-GPU Support

Two strategies are available for systems with multiple GPUs:

### Strategy 1: Separate GPUs per Pipeline

Load generation and edit pipelines on different GPUs so both can remain in VRAM:

```bash
python server.py --generation-device cuda:0 --edit-device cuda:1
```

| Pros | Cons |
|------|------|
| Both pipelines always ready | Requires 2+ GPUs with enough VRAM each |
| No swapping overhead | Each GPU must fit its full model |
| Simple to understand | |

**Best for:** Systems with 2+ mid-range GPUs (e.g., 2x RTX 3090)

### Strategy 2: Model Parallelism (device_map)

Distribute a single large model's layers across all available GPUs:

```bash
python server.py --device-map
```

| Pros | Cons |
|------|------|
| Can fit very large models | Cross-GPU communication overhead |
| Uses all available VRAM | Higher latency per inference |
| Automatic layer distribution | More complex debugging |

**Best for:** Running models that don't fit on a single GPU

### Combining Strategies

You can use `--device-map` with only one pipeline enabled:

```bash
# Distribute edit model across GPUs, disable generation
python server.py --device-map --disable-generation
```

### Checking GPU Setup

```python
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} - {torch.cuda.get_device_name(i)}")
    print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
```

## LoRA System

### Loading LoRAs

**From HuggingFace:**
1. Open "Manage LoRAs" panel
2. Select "HuggingFace" source
3. Enter repo ID (e.g., `lightx2v/Qwen-Image-Lightning`)
4. Optionally specify weight file name
5. Click "Load LoRA"

**From Local Files:**
1. Place `.safetensors` files in the `loras/` folder
2. Open "Manage LoRAs" panel
3. Select "Local File" source
4. Choose from available local LoRAs
5. Click "Load"

### Preset LoRAs

Edit `loras/presets.json` to configure quick-load presets:

```json
{
  "presets": [
    {
      "name": "Lightning",
      "source": "huggingface",
      "repo_id": "lightx2v/Qwen-Image-Lightning",
      "weight_name": "Qwen-Image-Lightning-8steps-V1.1.safetensors",
      "pipeline": "edit",
      "description": "8-step fast inference (for --edit-model original only)",
      "recommended_steps": 8
    }
  ]
}
```

### LoRA Controls

- **Activate/Deactivate** - Toggle LoRA without unloading
- **Scale** - Adjust LoRA strength (0.0 - 2.0)
- **Unload** - Remove LoRA from memory

## API Endpoints

### Image Operations
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Submit generation job |
| POST | `/edit` | Submit edit job (supports multiple images) |
| GET | `/status/{job_id}` | Get job status |
| GET | `/jobs` | List all jobs |
| GET | `/queue` | Get queue info |

### LoRA Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/lora/load` | Load a LoRA |
| POST | `/lora/unload/{name}` | Unload a LoRA |
| POST | `/lora/activate/{name}` | Activate a LoRA |
| POST | `/lora/deactivate/{name}` | Deactivate a LoRA |
| GET | `/lora/list` | List loaded LoRAs |
| GET | `/lora/available` | List local LoRA files |
| GET | `/lora/presets` | Get configured presets |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/system/info` | Server configuration and GPU stats |

## Project Structure

```
qwen-image-studio/
├── server.py              # FastAPI server
├── static/
│   └── index.html         # Web UI
├── loras/
│   ├── presets.json       # LoRA preset configuration
│   └── *.safetensors      # Local LoRA files
├── generated_images/      # Output images
└── uploaded_images/       # Input images
```

## Troubleshooting

### Out of Memory
- Enable `--pipeline-swap` to swap models between CPU/GPU
- Use `--quantize` with original edit model (note: incompatible with `--pipeline-swap`)
- Reduce `--max-pixels` for smaller image processing
- Disable unused pipeline with `--disable-generation` or `--disable-edit`

### Quantization Notes
- `--quantize` works with all models (generation and edit)
- Quantizes both transformer and text encoder to 4-bit NF4
- Quantized models are pinned to CUDA and cannot use `--pipeline-swap`
- Requires `bitsandbytes` package installed

### Multi-Image Not Working
Multi-image editing requires `--edit-model 2509` or `--edit-model 2511`. The original model only supports single images.

## License

MIT - See Qwen-Image model cards for model-specific licensing.

## Acknowledgments

- [Qwen-Image](https://github.com/QwenLM/Qwen-Image) by Alibaba
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
