"""
Tests for Pydantic request/response models.
"""
import pytest
from pydantic import ValidationError


class TestGenerationRequest:
    """Tests for GenerationRequest model."""

    def test_minimal_request(self):
        """Test creating a request with only required field."""
        from server import GenerationRequest
        req = GenerationRequest(prompt="A beautiful sunset")
        assert req.prompt == "A beautiful sunset"
        assert req.negative_prompt == ""
        assert req.width == 512
        assert req.height == 512
        assert req.num_inference_steps == 20
        assert req.cfg_scale == 4.0
        assert req.seed is None

    def test_full_request(self):
        """Test creating a request with all fields."""
        from server import GenerationRequest
        req = GenerationRequest(
            prompt="A beautiful sunset",
            negative_prompt="blurry, low quality",
            width=1024,
            height=768,
            num_inference_steps=50,
            cfg_scale=7.5,
            seed=42
        )
        assert req.prompt == "A beautiful sunset"
        assert req.negative_prompt == "blurry, low quality"
        assert req.width == 1024
        assert req.height == 768
        assert req.num_inference_steps == 50
        assert req.cfg_scale == 7.5
        assert req.seed == 42

    def test_missing_prompt_raises_error(self):
        """Test that missing prompt raises validation error."""
        from server import GenerationRequest
        with pytest.raises(ValidationError):
            GenerationRequest()


class TestEditRequest:
    """Tests for EditRequest model."""

    def test_minimal_request(self):
        """Test creating an edit request with only required field."""
        from server import EditRequest
        req = EditRequest(prompt="Make the sky purple")
        assert req.prompt == "Make the sky purple"
        assert req.negative_prompt == " "
        assert req.num_inference_steps == 50
        assert req.cfg_scale == 4.0
        assert req.seed is None

    def test_full_request(self):
        """Test creating an edit request with all fields."""
        from server import EditRequest
        req = EditRequest(
            prompt="Make the sky purple",
            negative_prompt="blurry",
            num_inference_steps=30,
            cfg_scale=5.0,
            seed=123
        )
        assert req.prompt == "Make the sky purple"
        assert req.negative_prompt == "blurry"
        assert req.num_inference_steps == 30
        assert req.cfg_scale == 5.0
        assert req.seed == 123


class TestJobResponse:
    """Tests for JobResponse model."""

    def test_job_response_creation(self):
        """Test creating a job response."""
        from server import JobResponse
        resp = JobResponse(
            job_id="test-123",
            status="queued",
            message="Job submitted"
        )
        assert resp.job_id == "test-123"
        assert resp.status == "queued"
        assert resp.message == "Job submitted"


class TestJobStatus:
    """Tests for JobStatus model."""

    def test_minimal_job_status(self):
        """Test creating a minimal job status."""
        from server import JobStatus, JobType
        status = JobStatus(
            job_id="test-456",
            job_type=JobType.GENERATE,
            status="queued",
            created_at="2024-01-01T00:00:00"
        )
        assert status.job_id == "test-456"
        assert status.job_type == JobType.GENERATE
        assert status.status == "queued"
        assert status.progress is None
        assert status.output_image_url is None

    def test_full_job_status(self):
        """Test creating a full job status with all fields."""
        from server import JobStatus, JobType
        status = JobStatus(
            job_id="test-789",
            job_type=JobType.EDIT,
            status="completed",
            progress=1.0,
            input_image_url="/uploads/input.png",
            input_image_urls=["/uploads/input1.png", "/uploads/input2.png"],
            output_image_url="/images/output.png",
            created_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:05:00",
            prompt="Edit the sky",
            width=512,
            height=512,
            num_inference_steps=50,
            cfg_scale=4.0
        )
        assert status.status == "completed"
        assert status.progress == 1.0
        assert len(status.input_image_urls) == 2


class TestJobType:
    """Tests for JobType enum."""

    def test_job_types(self):
        """Test JobType enum values."""
        from server import JobType
        assert JobType.GENERATE == "generate"
        assert JobType.EDIT == "edit"


class TestLoRAModels:
    """Tests for LoRA-related models."""

    def test_lora_source_enum(self):
        """Test LoRASource enum values."""
        from server import LoRASource
        assert LoRASource.huggingface == "huggingface"
        assert LoRASource.local == "local"

    def test_lora_request_huggingface(self):
        """Test LoRARequest for HuggingFace source."""
        from server import LoRARequest, LoRASource
        req = LoRARequest(
            name="test-lora",
            source=LoRASource.huggingface,
            repo_id="test/repo",
            scale=0.8,
            pipeline="edit"
        )
        assert req.name == "test-lora"
        assert req.source == LoRASource.huggingface
        assert req.repo_id == "test/repo"
        assert req.scale == 0.8
        assert req.pipeline == "edit"

    def test_lora_request_local(self):
        """Test LoRARequest for local source."""
        from server import LoRARequest, LoRASource
        req = LoRARequest(
            name="local-lora",
            source=LoRASource.local,
            weight_name="my_lora.safetensors",
            pipeline="generation"
        )
        assert req.source == LoRASource.local
        assert req.weight_name == "my_lora.safetensors"

    def test_lora_request_defaults(self):
        """Test LoRARequest default values."""
        from server import LoRARequest, LoRASource
        req = LoRARequest(name="default-lora")
        assert req.source == LoRASource.huggingface
        assert req.scale == 1.0
        assert req.pipeline == "edit"

    def test_lora_info(self):
        """Test LoRAInfo model."""
        from server import LoRAInfo, LoRASource
        info = LoRAInfo(
            name="my-lora",
            source=LoRASource.huggingface,
            repo_id="test/repo",
            scale=1.0,
            pipeline="edit",
            loaded_at="2024-01-01T00:00:00",
            active=True
        )
        assert info.name == "my-lora"
        assert info.active is True

    def test_lora_preset(self):
        """Test LoRAPreset model."""
        from server import LoRAPreset, LoRASource
        preset = LoRAPreset(
            name="Lightning",
            source=LoRASource.huggingface,
            repo_id="test/lightning",
            pipeline="edit",
            description="Fast 8-step generation",
            recommended_steps=8
        )
        assert preset.name == "Lightning"
        assert preset.recommended_steps == 8

    def test_lora_request_pipeline_validation(self):
        """Test that pipeline field only accepts valid values."""
        from server import LoRARequest
        # Valid values
        for pipeline in ["generation", "edit", "both"]:
            req = LoRARequest(name="test", pipeline=pipeline)
            assert req.pipeline == pipeline

        # Invalid value should raise error
        with pytest.raises(ValidationError):
            LoRARequest(name="test", pipeline="invalid")
