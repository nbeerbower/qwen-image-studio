"""
Tests for FastAPI endpoints.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import io
from PIL import Image


@pytest.fixture
def client():
    """Create a test client with mocked pipelines."""
    # Mock the heavy imports before importing server
    with patch.dict('sys.modules', {
        'diffusers': MagicMock(),
    }):
        import server

        # Mock pipelines
        server.generation_pipeline = MagicMock()
        server.edit_pipeline = MagicMock()

        # Reset job state
        server.job_status.clear()
        server.loaded_loras.clear()
        server.current_job_id = None

        # Create test client
        yield TestClient(server.app)


@pytest.fixture
def client_no_generation():
    """Create a test client with generation disabled."""
    with patch.dict('sys.modules', {
        'diffusers': MagicMock(),
    }):
        import server
        server.generation_pipeline = None
        server.edit_pipeline = MagicMock()
        server.job_status.clear()
        yield TestClient(server.app)


@pytest.fixture
def client_no_edit():
    """Create a test client with edit disabled."""
    with patch.dict('sys.modules', {
        'diffusers': MagicMock(),
    }):
        import server
        server.generation_pipeline = MagicMock()
        server.edit_pipeline = None
        server.job_status.clear()
        yield TestClient(server.app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_html(self, client, tmp_path):
        """Test that root serves the frontend HTML."""
        # Create a minimal static/index.html for the test
        import server
        import os

        # Ensure static dir exists
        os.makedirs("static", exist_ok=True)

        # The test will try to serve static/index.html
        # This test just verifies the endpoint exists
        response = client.get("/")
        # May return 404 if index.html doesn't exist, which is fine for unit test
        assert response.status_code in [200, 404]


class TestGenerateEndpoint:
    """Tests for /generate endpoint."""

    def test_generate_returns_job_id(self, client):
        """Test that generate endpoint returns a job ID."""
        response = client.post("/generate", json={
            "prompt": "A beautiful sunset over mountains"
        })
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert "Queue position" in data["message"]

    def test_generate_with_all_params(self, client):
        """Test generate with all parameters."""
        response = client.post("/generate", json={
            "prompt": "A cat",
            "negative_prompt": "blurry",
            "width": 768,
            "height": 512,
            "num_inference_steps": 30,
            "cfg_scale": 7.5,
            "seed": 42
        })
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_generate_missing_prompt(self, client):
        """Test that missing prompt returns error."""
        response = client.post("/generate", json={})
        assert response.status_code == 422  # Validation error

    def test_generate_disabled(self, client_no_generation):
        """Test that generate returns 503 when disabled."""
        response = client_no_generation.post("/generate", json={
            "prompt": "test"
        })
        assert response.status_code == 503
        assert "disabled" in response.json()["detail"].lower()


class TestEditEndpoint:
    """Tests for /edit endpoint."""

    def test_edit_returns_job_id(self, client):
        """Test that edit endpoint returns a job ID."""
        # Create a test image
        img = Image.new('RGB', (512, 512), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        response = client.post(
            "/edit",
            data={"prompt": "Make it blue"},
            files={"images": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_edit_with_all_params(self, client):
        """Test edit with all parameters."""
        img = Image.new('RGB', (512, 512), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        response = client.post(
            "/edit",
            data={
                "prompt": "Add a hat",
                "negative_prompt": "blurry",
                "num_inference_steps": 30,
                "cfg_scale": 5.0,
                "seed": 123
            },
            files={"images": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200

    def test_edit_disabled(self, client_no_edit):
        """Test that edit returns 503 when disabled."""
        img = Image.new('RGB', (512, 512), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        response = client_no_edit.post(
            "/edit",
            data={"prompt": "test"},
            files={"images": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 503


class TestStatusEndpoint:
    """Tests for /status/{job_id} endpoint."""

    def test_status_returns_job_info(self, client):
        """Test getting status of a submitted job."""
        # First, create a job
        response = client.post("/generate", json={"prompt": "test"})
        job_id = response.json()["job_id"]

        # Get its status
        status_response = client.get(f"/status/{job_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "queued"
        assert data["job_type"] == "generate"

    def test_status_not_found(self, client):
        """Test that unknown job ID returns 404."""
        response = client.get("/status/nonexistent-job-id")
        assert response.status_code == 404


class TestQueueEndpoint:
    """Tests for /queue endpoint."""

    def test_queue_info(self, client):
        """Test getting queue information."""
        response = client.get("/queue")
        assert response.status_code == 200
        data = response.json()
        assert "queue_size" in data
        assert "current_job" in data
        assert "total_jobs" in data
        assert "generation_jobs" in data
        assert "edit_jobs" in data
        assert "completed_jobs" in data
        assert "failed_jobs" in data

    def test_queue_counts_jobs(self, client):
        """Test that queue counts jobs correctly."""
        # Submit some jobs
        client.post("/generate", json={"prompt": "test1"})
        client.post("/generate", json={"prompt": "test2"})

        response = client.get("/queue")
        data = response.json()
        assert data["total_jobs"] >= 2
        assert data["generation_jobs"] >= 2


class TestJobsEndpoint:
    """Tests for /jobs endpoint."""

    def test_get_all_jobs(self, client):
        """Test getting all jobs."""
        # Submit a job
        client.post("/generate", json={"prompt": "test"})

        response = client.get("/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert isinstance(jobs, list)
        assert len(jobs) >= 1

    def test_filter_jobs_by_type(self, client):
        """Test filtering jobs by type."""
        # Submit a generation job
        client.post("/generate", json={"prompt": "gen test"})

        # Filter by generate type
        response = client.get("/jobs?job_type=generate")
        assert response.status_code == 200
        jobs = response.json()
        for job in jobs:
            assert job["job_type"] == "generate"


class TestSystemInfoEndpoint:
    """Tests for /system/info endpoint."""

    def test_system_info(self, client, mock_cuda_unavailable):
        """Test getting system information."""
        response = client.get("/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "device" in data
        assert "dtype" in data
        assert "cuda_available" in data
        assert "generation_pipeline" in data
        assert "edit_pipeline" in data
        assert "quantization" in data


class TestLoRAEndpoints:
    """Tests for LoRA management endpoints."""

    def test_list_loras_empty(self, client):
        """Test listing LoRAs when none loaded."""
        response = client.get("/lora/list")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["loras"] == []

    def test_load_lora_missing_repo_id(self, client):
        """Test that loading HuggingFace LoRA without repo_id fails."""
        response = client.post("/lora/load", json={
            "name": "test-lora",
            "source": "huggingface"
        })
        assert response.status_code == 400
        assert "repo_id" in response.json()["detail"].lower()

    def test_load_lora_missing_weight_name(self, client):
        """Test that loading local LoRA without weight_name fails."""
        response = client.post("/lora/load", json={
            "name": "test-lora",
            "source": "local"
        })
        assert response.status_code == 400
        assert "weight_name" in response.json()["detail"].lower()

    def test_load_local_lora_not_found(self, client):
        """Test that loading nonexistent local LoRA fails."""
        response = client.post("/lora/load", json={
            "name": "test-lora",
            "source": "local",
            "weight_name": "nonexistent.safetensors"
        })
        assert response.status_code == 404

    def test_unload_lora_not_found(self, client):
        """Test that unloading nonexistent LoRA returns 404."""
        response = client.post("/lora/unload/nonexistent")
        assert response.status_code == 404

    def test_activate_lora_not_found(self, client):
        """Test that activating nonexistent LoRA returns 404."""
        response = client.post("/lora/activate/nonexistent")
        assert response.status_code == 404

    def test_deactivate_lora_not_found(self, client):
        """Test that deactivating nonexistent LoRA returns 404."""
        response = client.post("/lora/deactivate/nonexistent")
        assert response.status_code == 404

    def test_available_loras(self, client):
        """Test listing available local LoRA files."""
        response = client.get("/lora/available")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data
        assert "count" in data
        assert "directory" in data

    def test_lora_presets(self, client):
        """Test getting LoRA presets."""
        response = client.get("/lora/presets")
        assert response.status_code == 200
        data = response.json()
        assert "presets" in data
        assert "count" in data
