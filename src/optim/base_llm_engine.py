import logging
import os

import litellm
from PIL import Image
import io
import base64

# litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
os.environ["LITELLM_LOG"] = "DEBUG"
logger = logging.getLogger(__name__)


class BaseLLMEngine:
    """LiteLLM-based prompt optimizer using the official LiteLLM library."""

    def __init__(
        self,
        llm_engine: str = "azure/gpt-4o",
        **kwargs,
    ):
        """Initialize the LiteLLM optimizer."""
        self._setup_llm_engine(llm_engine)

    def _setup_llm_engine(self, llm_engine: str) -> None:
        """Setup the LiteLLM engine."""
        try:
            self.engine = litellm.completion
            self.model = llm_engine
            logger.info(f"LiteLLM initialized for {llm_engine}")

        except Exception as e:
            logger.error(f"Failed to setup LiteLLM engine: {e}")
            raise RuntimeError(
                f"LiteLLM engine setup failed for {llm_engine}: {e}"
            )

    def _image_to_base64_url(
        self, image: Image.Image, max_size_mb: float = 15.0
    ) -> str:
        """
        Convert PIL Image to base64 data URL for LLM vision input with size compression.

        Args:
            image: PIL Image to convert
            max_size_mb: Maximum size in MB (default 15MB to stay under 20MB limit)

        Returns:
            Base64 data URL string
        """
        # First, try with original size but optimize format and quality
        compressed_image = self._compress_image_for_api(image, max_size_mb)

        buffer = io.BytesIO()
        # Use JPEG for better compression, PNG for images with transparency
        if (
            compressed_image.mode in ("RGBA", "LA")
            or "transparency" in compressed_image.info
        ):
            compressed_image.save(buffer, format="PNG", optimize=True)
            format_str = "png"
        else:
            # Convert to RGB if needed for JPEG
            if compressed_image.mode != "RGB":
                compressed_image = compressed_image.convert("RGB")
            compressed_image.save(
                buffer, format="JPEG", quality=85, optimize=True
            )
            format_str = "jpeg"

        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Log the final size for debugging
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        # logger.info(f"Compressed image to {size_mb:.2f}MB ({format_str.upper()})")

        return f"data:image/{format_str};base64,{image_base64}"

    def _compress_image_for_api(
        self, image: Image.Image, max_size_mb: float
    ) -> Image.Image:
        """
        Compress image to fit within size limits while maintaining visual quality.

        Args:
            image: Original PIL Image
            max_size_mb: Maximum size in MB

        Returns:
            Compressed PIL Image
        """
        max_size_bytes = max_size_mb * 1024 * 1024

        # Start with original image
        current_image = image.copy()

        # First check if image is already small enough
        buffer = io.BytesIO()
        if (
            current_image.mode in ("RGBA", "LA")
            or "transparency" in current_image.info
        ):
            current_image.save(buffer, format="PNG", optimize=True)
        else:
            if current_image.mode != "RGB":
                current_image = current_image.convert("RGB")
            current_image.save(buffer, format="JPEG", quality=85, optimize=True)

        if len(buffer.getvalue()) <= max_size_bytes:
            return current_image

        logger.info(
            f"Image too large ({len(buffer.getvalue()) / (1024 * 1024):.2f}MB), compressing..."
        )

        # Progressive resize until we fit the size limit
        scale_factor = 0.9
        max_attempts = 10
        attempt = 0

        while (
            len(buffer.getvalue()) > max_size_bytes and attempt < max_attempts
        ):
            attempt += 1

            # Resize image
            new_width = int(current_image.width * scale_factor)
            new_height = int(current_image.height * scale_factor)
            current_image = current_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

            # Test new size
            buffer = io.BytesIO()
            if (
                current_image.mode in ("RGBA", "LA")
                or "transparency" in current_image.info
            ):
                current_image.save(buffer, format="PNG", optimize=True)
            else:
                if current_image.mode != "RGB":
                    current_image = current_image.convert("RGB")
                current_image.save(
                    buffer, format="JPEG", quality=85, optimize=True
                )

            current_size_mb = len(buffer.getvalue()) / (1024 * 1024)
            logger.info(
                f"Attempt {attempt}: Resized to {new_width}x{new_height}, size: {current_size_mb:.2f}MB"
            )

        if len(buffer.getvalue()) > max_size_bytes:
            logger.warning(
                f"Could not compress image below {max_size_mb}MB limit after {max_attempts} attempts"
            )

        return current_image

    def __call__(self):
        pass

    def _create_prompt(self):
        pass

    def _parse_response(self):
        pass
