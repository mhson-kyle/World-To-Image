import torch
import torch.nn as nn
import clip
from PIL import Image
from typing import Dict, Optional, Any, List
import numpy as np
import os
import pytorch_lightning as pl

from .base import BaseScorer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MLP(pl.LightningModule):
    """
    Multi-layer perceptron for aesthetic scoring.

    This is a pre-trained model that takes CLIP image embeddings and outputs
    aesthetic quality scores.
    """

    def __init__(
        self, input_size: int, xcol: str = "emb", ycol: str = "avg_rating"
    ):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    """Normalize numpy array along specified axis."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def normalize_tensor(
    a: torch.Tensor, axis: int = -1, order: int = 2
) -> torch.Tensor:
    """Normalize torch tensor along specified axis."""
    norm_sq = a.abs().pow(order).sum(dim=axis, keepdim=True)
    l2 = norm_sq.pow(1.0 / order)
    l2 = torch.where(l2 == 0, torch.ones_like(l2), l2)
    return a / l2


class AestheticScorer(BaseScorer):
    """
    Aesthetic scorer for image quality evaluation.

    This scorer uses a pre-trained MLP model on CLIP embeddings to evaluate
    the aesthetic quality of images independently of text prompts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the aesthetic scorer.

        Args:
            config: Configuration dictionary for the scorer
        """
        super().__init__(config)

        # Default configuration
        default_config = {
            "aesthetic_model_path": "src/scoring/sac+logos+ava1-l14-linearMSE.pth",
            "clip_model_name": "ViT-L/14",
            "normalize_scores": True,
            "temperature": 1.0,
        }
        self.config.update(default_config)
        if config:
            self.config.update(config)

        self.aesthetic_model_path = self.config["aesthetic_model_path"]
        self.clip_model_name = self.config["clip_model_name"]

        # Model components
        self._clip_model = None
        self._clip_preprocess = None
        self._aesthetic_model = None

        logger.debug(
            f"AestheticScorer initialized with CLIP model: {self.clip_model_name}"
        )

    def load_model(self) -> None:
        """
        Load the CLIP model and aesthetic MLP model.

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Load CLIP model
            self._clip_model, self._clip_preprocess = clip.load(
                self.clip_model_name, device=self.device
            )
            self._clip_model.eval()

            # Load aesthetic MLP model
            if not os.path.exists(self.aesthetic_model_path):
                raise FileNotFoundError(
                    f"Aesthetic model not found at: {self.aesthetic_model_path}"
                )

            self._aesthetic_model = MLP(input_size=768)
            state_dict = torch.load(
                self.aesthetic_model_path, map_location=self.device
            )
            self._aesthetic_model.load_state_dict(state_dict)
            self._aesthetic_model.to(self.device)
            self._aesthetic_model.eval()

            logger.debug(
                f"Aesthetic models loaded successfully on {self.device}"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load aesthetic models: {str(e)}"
            ) from e

    def score(self, image: Image.Image, prompt: str = "", **kwargs) -> float:
        """
        Score an image for aesthetic quality.

        Args:
            image: Input image
            prompt: Input text prompt (ignored for aesthetic scoring)
            **kwargs: Additional arguments

        Returns:
            Aesthetic quality score (higher is better)
        """
        if not self.is_ready():
            self.load_model()

        try:
            # Preprocess image using CLIP preprocessing
            img_tensor = (
                self._clip_preprocess(image).unsqueeze(0).to(self.device)
            )

            # Get CLIP image features
            with torch.no_grad():
                image_features = self._clip_model.encode_image(img_tensor)
                image_features = image_features.cpu().numpy()
                image_features = normalized(image_features)

            # Get aesthetic score from MLP
            with torch.no_grad():
                emb_tensor = (
                    torch.from_numpy(image_features).to(self.device).float()
                )
                score_tensor = self._aesthetic_model(emb_tensor)
                aesthetic_score = float(score_tensor.item())

            # Apply temperature scaling if specified
            temperature = kwargs.get(
                "temperature", self.config.get("temperature", 1.0)
            )
            if temperature != 1.0:
                aesthetic_score = aesthetic_score / temperature

            # Normalize if requested
            if self.config.get("normalize_scores", True):
                # The aesthetic model typically outputs scores in a specific range
                # We can normalize to [0, 1] or keep the original scale
                # For now, we'll keep the original scale as it's more meaningful
                pass

            return aesthetic_score

        except Exception as e:
            raise RuntimeError(f"Aesthetic scoring failed: {str(e)}") from e

    def score_batch(
        self,
        images: List[Image.Image],
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Score multiple images for aesthetic quality.

        Args:
            images: List of input images
            prompts: List of input prompts (ignored for aesthetic scoring)
            **kwargs: Additional arguments

        Returns:
            List of aesthetic quality scores
        """
        if not self.is_ready():
            self.load_model()

        try:
            scores = []
            temperature = kwargs.get(
                "temperature", self.config.get("temperature", 1.0)
            )

            for image in images:
                # Preprocess image
                img_tensor = (
                    self._clip_preprocess(image).unsqueeze(0).to(self.device)
                )

                # Get CLIP features
                with torch.no_grad():
                    image_features = self._clip_model.encode_image(img_tensor)
                    image_features = image_features.cpu().numpy()
                    image_features = normalized(image_features)

                # Get aesthetic score
                with torch.no_grad():
                    emb_tensor = (
                        torch.from_numpy(image_features).to(self.device).float()
                    )
                    score_tensor = self._aesthetic_model(emb_tensor)
                    aesthetic_score = float(score_tensor.item())

                # Apply temperature scaling
                if temperature != 1.0:
                    aesthetic_score = aesthetic_score / temperature

                scores.append(aesthetic_score)

            return scores

        except Exception as e:
            raise RuntimeError(
                f"Aesthetic batch scoring failed: {str(e)}"
            ) from e

    def analyze_aesthetic_components(
        self, image: Image.Image, **kwargs
    ) -> Dict[str, float]:
        """
        Analyze different aesthetic components of an image.

        Note: This method provides a simplified analysis since the MLP model
        outputs a single aesthetic score. For detailed component analysis,
        you would need a different model architecture.

        Args:
            image: Input image
            **kwargs: Additional arguments

        Returns:
            Dictionary with aesthetic component scores
        """
        if not self.is_ready():
            self.load_model()

        try:
            # Get overall aesthetic score
            overall_score = self.score(image, **kwargs)

            # For now, we'll return the overall score for all components
            # since the MLP model doesn't provide component breakdowns
            component_scores = {
                "composition": overall_score,
                "color": overall_score,
                "quality": overall_score,
                "artistic": overall_score,
                "beauty": overall_score,
                "overall": overall_score,
            }

            return component_scores

        except Exception as e:
            raise RuntimeError(
                f"Aesthetic component analysis failed: {str(e)}"
            ) from e

    def set_device(self, device: str) -> None:
        """
        Set the device for model execution.

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
        """
        super().set_device(device)

        # Move models to device if they're loaded
        if self._clip_model is not None:
            self._clip_model = self._clip_model.to(self.device)
        if self._aesthetic_model is not None:
            self._aesthetic_model = self._aesthetic_model.to(self.device)

    def is_ready(self) -> bool:
        """
        Check if the scorer is ready for use.

        Returns:
            True if all models are loaded and ready
        """
        return (
            self._clip_model is not None
            and self._clip_preprocess is not None
            and self._aesthetic_model is not None
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"AestheticScorer(clip_model={self.clip_model_name}, device={self.device}, ready={self.is_ready()})"
