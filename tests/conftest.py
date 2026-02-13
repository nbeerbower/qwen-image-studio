"""
Pytest configuration and fixtures for Qwen Image Studio tests.
"""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path so we can import server module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def mock_torch():
    """Mock torch module to avoid GPU requirements in tests."""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = False
    mock_cuda.device_count.return_value = 0

    with patch.dict('sys.modules', {
        'torch.cuda': mock_cuda,
    }):
        yield


@pytest.fixture
def mock_cuda_available():
    """Fixture that simulates CUDA being available."""
    import torch
    original_is_available = torch.cuda.is_available
    original_device_count = torch.cuda.device_count

    torch.cuda.is_available = MagicMock(return_value=True)
    torch.cuda.device_count = MagicMock(return_value=2)

    yield

    torch.cuda.is_available = original_is_available
    torch.cuda.device_count = original_device_count


@pytest.fixture
def mock_cuda_unavailable():
    """Fixture that simulates CUDA being unavailable."""
    import torch
    original_is_available = torch.cuda.is_available
    original_device_count = torch.cuda.device_count

    torch.cuda.is_available = MagicMock(return_value=False)
    torch.cuda.device_count = MagicMock(return_value=0)

    yield

    torch.cuda.is_available = original_is_available
    torch.cuda.device_count = original_device_count


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    from PIL import Image
    return Image.new('RGB', (1024, 768), color='red')


@pytest.fixture
def large_image():
    """Create a large PIL Image that exceeds default max_pixels."""
    from PIL import Image
    # 2000x2000 = 4,000,000 pixels (exceeds default 1,048,576)
    return Image.new('RGB', (2000, 2000), color='blue')


@pytest.fixture
def small_image():
    """Create a small PIL Image."""
    from PIL import Image
    return Image.new('RGB', (256, 256), color='green')


@pytest.fixture
def temp_loras_dir(tmp_path):
    """Create a temporary loras directory with test files."""
    loras_dir = tmp_path / "loras"
    loras_dir.mkdir()

    # Create a fake safetensors file
    fake_lora = loras_dir / "test_lora.safetensors"
    fake_lora.write_bytes(b"fake safetensors content")

    # Create presets.json
    presets_file = loras_dir / "presets.json"
    presets_file.write_text('''{
        "presets": [
            {
                "name": "Test Preset",
                "source": "huggingface",
                "repo_id": "test/repo",
                "pipeline": "edit",
                "description": "A test preset"
            }
        ]
    }''')

    return loras_dir


@pytest.fixture
def mock_args():
    """Create a mock args namespace for testing."""
    from argparse import Namespace
    return Namespace(
        host="0.0.0.0",
        port=8000,
        edit_model="2511",
        generation_model="2512",
        quantize=False,
        cpu_offload=True,
        max_pixels=1048576,
        disable_generation=False,
        disable_edit=False,
        device="auto",
        generation_device=None,
        edit_device=None,
        device_map=False,
        pipeline_swap=False,
        keep_in_vram=None
    )
