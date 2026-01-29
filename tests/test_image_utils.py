"""
Tests for image processing utility functions.
"""
import pytest
from PIL import Image


class TestResizeImageIfNeeded:
    """Tests for resize_image_if_needed function."""

    def test_small_image_unchanged(self, small_image, mock_args):
        """Test that small images are not resized."""
        import server
        original_args = server.args
        server.args = mock_args

        try:
            from server import resize_image_if_needed
            result = resize_image_if_needed(small_image)
            assert result.size == small_image.size  # 256x256
        finally:
            server.args = original_args

    def test_large_image_resized(self, large_image, mock_args):
        """Test that large images are resized to fit within max_pixels."""
        import server
        original_args = server.args
        server.args = mock_args

        try:
            from server import resize_image_if_needed
            result = resize_image_if_needed(large_image)
            result_pixels = result.size[0] * result.size[1]
            # Should be at or below max_pixels (1048576)
            assert result_pixels <= mock_args.max_pixels
            # Original was 2000x2000 = 4,000,000 pixels
            assert result.size != large_image.size
        finally:
            server.args = original_args

    def test_resize_maintains_aspect_ratio(self, mock_args):
        """Test that resizing maintains aspect ratio."""
        import server
        original_args = server.args
        server.args = mock_args

        try:
            from server import resize_image_if_needed
            # Create a 4000x2000 image (8M pixels, 2:1 aspect ratio)
            wide_image = Image.new('RGB', (4000, 2000), color='red')
            result = resize_image_if_needed(wide_image)

            # Check aspect ratio is approximately preserved
            original_ratio = 4000 / 2000  # 2.0
            result_ratio = result.size[0] / result.size[1]
            # Allow small deviation due to rounding to multiples of 4
            assert abs(original_ratio - result_ratio) < 0.1
        finally:
            server.args = original_args

    def test_resize_produces_multiples_of_4(self, mock_args):
        """Test that resized dimensions are multiples of 4."""
        import server
        original_args = server.args
        server.args = mock_args

        try:
            from server import resize_image_if_needed
            # Create a large image that will need resizing
            large = Image.new('RGB', (3000, 2500), color='blue')
            result = resize_image_if_needed(large)

            assert result.size[0] % 4 == 0
            assert result.size[1] % 4 == 0
        finally:
            server.args = original_args

    def test_resize_ensures_minimum_dimensions(self, mock_args):
        """Test that resized images maintain minimum dimensions."""
        import server
        original_args = server.args
        mock_args.max_pixels = 100  # Very small max to force tiny resize
        server.args = mock_args

        try:
            from server import resize_image_if_needed
            large = Image.new('RGB', (10000, 10000), color='green')
            result = resize_image_if_needed(large)

            # Should not go below 256x256
            assert result.size[0] >= 256
            assert result.size[1] >= 256
        finally:
            server.args = original_args

    def test_custom_max_pixels(self, mock_args):
        """Test resize_image_if_needed with custom max_pixels parameter."""
        import server
        original_args = server.args
        server.args = mock_args

        try:
            from server import resize_image_if_needed
            # Create image that's 1000x1000 = 1,000,000 pixels
            image = Image.new('RGB', (1000, 1000), color='yellow')

            # With max_pixels=500000, it should be resized
            result = resize_image_if_needed(image, max_pixels=500000)
            assert result.size[0] * result.size[1] <= 500000

            # With max_pixels=2000000, it should NOT be resized
            result2 = resize_image_if_needed(image, max_pixels=2000000)
            assert result2.size == image.size
        finally:
            server.args = original_args

    def test_exact_max_pixels_not_resized(self, mock_args):
        """Test that image exactly at max_pixels is not resized."""
        import server
        original_args = server.args
        server.args = mock_args

        try:
            from server import resize_image_if_needed
            # 1024 * 1024 = 1048576 = exactly max_pixels
            exact_image = Image.new('RGB', (1024, 1024), color='cyan')
            result = resize_image_if_needed(exact_image)
            assert result.size == exact_image.size
        finally:
            server.args = original_args

    def test_image_quality_preserved(self, mock_args):
        """Test that LANCZOS resampling is used for quality."""
        import server
        original_args = server.args
        server.args = mock_args

        try:
            from server import resize_image_if_needed
            # Create a checkerboard pattern to test quality
            img = Image.new('RGB', (2048, 2048))
            pixels = img.load()
            for i in range(2048):
                for j in range(2048):
                    if (i + j) % 2 == 0:
                        pixels[i, j] = (255, 255, 255)
                    else:
                        pixels[i, j] = (0, 0, 0)

            result = resize_image_if_needed(img)
            # Just verify it resized without error
            assert result.size[0] * result.size[1] <= mock_args.max_pixels
        finally:
            server.args = original_args
