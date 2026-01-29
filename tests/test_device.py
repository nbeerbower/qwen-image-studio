"""
Tests for device detection and parsing functions.
"""
import pytest
from unittest.mock import patch, MagicMock
import torch


class TestParseDevice:
    """Tests for parse_device function."""

    def test_cpu_device(self):
        """Test parsing CPU device string."""
        from server import parse_device
        device, dtype = parse_device("cpu")
        assert device == "cpu"
        assert dtype == torch.float32

    def test_auto_device_without_cuda(self, mock_cuda_unavailable):
        """Test auto device selection when CUDA is unavailable."""
        from server import parse_device
        device, dtype = parse_device("auto")
        assert device == "cpu"
        assert dtype == torch.float32

    def test_auto_device_with_cuda(self, mock_cuda_available):
        """Test auto device selection when CUDA is available."""
        from server import parse_device
        device, dtype = parse_device("auto")
        assert device == "cuda"
        assert dtype == torch.bfloat16

    def test_cuda_device_without_availability(self, mock_cuda_unavailable, capsys):
        """Test that CUDA fallback to CPU when not available."""
        from server import parse_device
        device, dtype = parse_device("cuda")
        assert device == "cpu"
        assert dtype == torch.float32
        captured = capsys.readouterr()
        assert "CUDA not available" in captured.out

    def test_cuda_device_with_availability(self, mock_cuda_available):
        """Test CUDA device when available."""
        from server import parse_device
        device, dtype = parse_device("cuda")
        assert device == "cuda"
        assert dtype == torch.bfloat16

    def test_cuda_specific_device(self, mock_cuda_available):
        """Test parsing specific CUDA device."""
        from server import parse_device
        device, dtype = parse_device("cuda:0")
        assert device == "cuda:0"
        assert dtype == torch.bfloat16

    def test_cuda_device_out_of_range(self, mock_cuda_available, capsys):
        """Test CUDA device index out of range falls back to cuda:0."""
        from server import parse_device
        device, dtype = parse_device("cuda:10")
        assert device == "cuda:0"
        assert dtype == torch.bfloat16
        captured = capsys.readouterr()
        assert "only" in captured.out and "GPUs available" in captured.out


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_uses_args(self, mock_args, mock_cuda_unavailable):
        """Test that get_device uses the args.device setting."""
        import server
        original_args = server.args
        server.args = mock_args
        mock_args.device = "cpu"

        try:
            device, dtype = server.get_device()
            assert device == "cpu"
            assert dtype == torch.float32
        finally:
            server.args = original_args


class TestGetGenerationDevice:
    """Tests for get_generation_device function."""

    def test_uses_generation_device_override(self, mock_args, mock_cuda_available):
        """Test that generation_device override works."""
        import server
        original_args = server.args
        mock_args.generation_device = "cuda:1"
        server.args = mock_args

        try:
            device, dtype = server.get_generation_device()
            assert device == "cuda:1"
            assert dtype == torch.bfloat16
        finally:
            server.args = original_args

    def test_falls_back_to_default_device(self, mock_args, mock_cuda_unavailable):
        """Test fallback to default device when no override."""
        import server
        original_args = server.args
        mock_args.generation_device = None
        mock_args.device = "cpu"
        server.args = mock_args

        try:
            device, dtype = server.get_generation_device()
            assert device == "cpu"
        finally:
            server.args = original_args


class TestGetEditDevice:
    """Tests for get_edit_device function."""

    def test_uses_edit_device_override(self, mock_args, mock_cuda_available):
        """Test that edit_device override works."""
        import server
        original_args = server.args
        mock_args.edit_device = "cuda:0"
        server.args = mock_args

        try:
            device, dtype = server.get_edit_device()
            assert device == "cuda:0"
            assert dtype == torch.bfloat16
        finally:
            server.args = original_args

    def test_falls_back_to_default_device(self, mock_args, mock_cuda_unavailable):
        """Test fallback to default device when no override."""
        import server
        original_args = server.args
        mock_args.edit_device = None
        mock_args.device = "cpu"
        server.args = mock_args

        try:
            device, dtype = server.get_edit_device()
            assert device == "cpu"
        finally:
            server.args = original_args


class TestGetGpuCount:
    """Tests for get_gpu_count function."""

    def test_gpu_count_with_cuda(self, mock_cuda_available):
        """Test GPU count when CUDA available."""
        from server import get_gpu_count
        count = get_gpu_count()
        assert count == 2  # Our mock returns 2

    def test_gpu_count_without_cuda(self, mock_cuda_unavailable):
        """Test GPU count when CUDA unavailable."""
        from server import get_gpu_count
        count = get_gpu_count()
        assert count == 0
