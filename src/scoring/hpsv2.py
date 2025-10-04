import logging
from typing import Dict, Optional, Any
from PIL import Image

from .base import BaseScorer

logger = logging.getLogger(__name__)


class HPSv2Scorer(BaseScorer):
    """
    HPSv2 scorer for human preference evaluation.

    HPSv2 is a CLIP-based human preference score model that provides
    accurate assessments of generated images based on human preferences.
    It's much smaller and more memory-efficient than HPSv3.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the HPSv2 scorer.

        Args:
            config: Configuration dictionary with keys:
                - device: Device to run the model on ('cuda', 'cpu', or specific GPU)
                - hps_version: HPSv2 model version ('v2' or 'v2.1', default: 'v2.1')
        """
        super().__init__(config)

        # HPSv2 specific configuration
        self.hps_version = self.config.get(
            "hps_version", "v2.1"
        )  # Default to v2.1 for better performance

        # Initialize model placeholder
        self._model_loaded = False

        logger.info(
            f"HPSv2 scorer initialized with version: {self.hps_version}"
        )

    def load_model(self) -> None:
        """
        Load the HPSv2 model.

        HPSv2 models are loaded on-demand when scoring, so this is a no-op.
        """
        try:
            # Import HPSv2 here to avoid dependency issues during import
            import hpsv2

            # Test that the module is working
            logger.info("✓ HPSv2 module loaded successfully")
            self._model_loaded = True

        except ImportError as e:
            raise RuntimeError(
                "HPSv2 library not found. Please install it with: pip install hpsv2"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load HPSv2 module: {str(e)}") from e

    def score(self, image: Image.Image, prompt: str, **kwargs) -> float:
        """
        Score the image using HPSv2 human preference model.

        Args:
            image: The image to score
            prompt: The text prompt to compare against
            **kwargs: Additional scoring parameters

        Returns:
            HPSv2 preference score (higher values indicate better human preference)

        Raises:
            RuntimeError: If scoring fails or model not loaded
        """
        if not self._model_loaded:
            self.load_model()

        try:
            # Import HPSv2 here to avoid dependency issues during import
            import hpsv2

            # HPSv2.score() can handle PIL images directly
            # It expects: hpsv2.score(imgs_path, prompt, hps_version="v2.1")
            score = hpsv2.score(image, prompt, hps_version=self.hps_version)

            # HPSv2 returns a list for single image scoring, extract the first element
            if isinstance(score, list):
                score = score[0]

            # Convert to float and handle numpy types
            if hasattr(score, "item"):
                score = score.item()

            return float(score)

        except Exception as e:
            logger.error(f"HPSv2 scoring failed: {str(e)}")
            raise RuntimeError(f"HPSv2 scoring failed: {str(e)}") from e

    def get_score_info(self) -> Dict[str, Any]:
        """
        Get information about the HPSv2 scoring method.

        Returns:
            Dictionary with scoring method information
        """
        return {
            "name": "HPSv2",
            "version": self.hps_version,
            "description": "Human Preference Score v2 - CLIP-based preference model",
            "paper": "Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis",
            "range": "Typically 20-35, higher is better",
            "model_type": "CLIP-based",
            "device": str(self.device),
            "memory_efficient": True,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"HPSv2Scorer(version={self.hps_version}, device={self.device}, ready={self._model_loaded})"
