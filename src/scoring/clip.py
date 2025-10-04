import torch
import clip
from PIL import Image
from typing import Dict, Optional, Any, List

from .base import BaseScorer


def get_preprocess(n_px=224):
    """Get CLIP preprocessing transforms."""
    from torchvision.transforms import (
        Compose,
        Resize,
        CenterCrop,
        ToTensor,
        Normalize,
    )

    return Compose(
        [
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class CLIPScorer(BaseScorer):
    """
    CLIP-based image-prompt alignment scorer.

    This class provides a concrete implementation of the BaseScorer interface
    using the official CLIP library to evaluate how well an image
    aligns with a given text prompt.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CLIP scorer.

        Args:
            config: Configuration dictionary for the scorer
        """
        super().__init__(config)
        self._processor = None
        self.model_id = "openai/clip-vit-large-patch14"

        # Set default configuration
        default_config = {
            "clip_model_name": "ViT-L/14",
            "normalize_scores": False,  # CLIP scores are typically in [-100, 100] range
            "temperature": 1.0,
            "negative": False,
        }
        self.config.update(default_config)
        if config:
            self.config.update(config)

        self.clip_model_name = self.config["clip_model_name"]
        self._model = None
        self._preprocess = None

        print(f"CLIPScorer initialized with model: {self.clip_model_name}")

    def load_model(self) -> None:
        """
        Load the CLIP model and preprocessing.

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            self._model, self._preprocess = clip.load(
                self.clip_model_name, device=self.device
            )
            self._model.eval()

            # print("CLIP model loaded.")

        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {str(e)}") from e

    def score(self, image: Image.Image, prompt: str, **kwargs) -> float:
        """
        Score the alignment between an image and a prompt using CLIP.

        Args:
            image: The image to score
            prompt: The text prompt to compare against
            **kwargs: Additional scoring parameters

        Returns:
            CLIP similarity score (typically in [-100, 100] range)

        Raises:
            RuntimeError: If scoring fails or model not loaded
        """
        if not self.is_ready():
            self.load_model()

        try:
            # Preprocess image
            image_tensor = self._preprocess(image).unsqueeze(0).to(self.device)

            # Tokenize text
            text_tokens = clip.tokenize([prompt], truncate=True).to(self.device)

            # Get embeddings
            with torch.no_grad():
                if self.device == "cuda":
                    image_tensor = image_tensor.half()

                image_features = self._model.encode_image(image_tensor)
                text_features = self._model.encode_text(text_tokens)

                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

                # Compute similarity using dot product
                similarity = (image_features * text_features).sum(dim=-1).item()

                # Apply negative flag if specified
                negative = kwargs.get(
                    "negative", self.config.get("negative", False)
                )
                if negative:
                    similarity = -similarity

                # Scale to match the example implementation
                clipscore = float(similarity * 100.0)

                # Apply temperature scaling if specified
                temperature = kwargs.get(
                    "temperature", self.config.get("temperature", 1.0)
                )
                if temperature != 1.0:
                    clipscore = clipscore / temperature

                return clipscore

        except Exception as e:
            raise RuntimeError(f"CLIP scoring failed: {str(e)}") from e

    def score_batch(
        self, images: List[Image.Image], prompts: List[str], **kwargs
    ) -> List[float]:
        """
        Score multiple image-prompt pairs efficiently.

        Args:
            images: List of images to score
            prompts: List of text prompts to compare against
            **kwargs: Additional scoring parameters

        Returns:
            List of CLIP similarity scores

        Raises:
            RuntimeError: If scoring fails or model not loaded
            ValueError: If number of images and prompts don't match
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images and prompts must match")

        if not self.is_ready():
            self.load_model()

        try:
            scores = []
            temperature = kwargs.get(
                "temperature", self.config.get("temperature", 1.0)
            )
            negative = kwargs.get(
                "negative", self.config.get("negative", False)
            )

            for image, prompt in zip(images, prompts):
                # Preprocess image
                image_tensor = (
                    self._preprocess(image).unsqueeze(0).to(self.device)
                )

                # Tokenize text
                text_tokens = clip.tokenize([prompt], truncate=True).to(
                    self.device
                )

                # Get embeddings
                with torch.no_grad():
                    if self.device == "cuda":
                        image_tensor = image_tensor.half()

                    image_features = self._model.encode_image(image_tensor)
                    text_features = self._model.encode_text(text_tokens)

                    # Normalize features
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )

                    # Compute similarity
                    similarity = (
                        (image_features * text_features).sum(dim=-1).item()
                    )

                    # Apply negative flag
                    if negative:
                        similarity = -similarity

                    # Scale to match the example implementation
                    clipscore = float(similarity * 100.0)

                    # Apply temperature scaling
                    if temperature != 1.0:
                        clipscore = clipscore / temperature

                    scores.append(clipscore)

            return scores

        except Exception as e:
            raise RuntimeError(f"CLIP batch scoring failed: {str(e)}") from e

    def set_device(self, device: str) -> None:
        """
        Set the device for model execution.

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
        """
        super().set_device(device)

        # Move model to device if it's loaded
        if self._model is not None:
            self._model = self._model.to(self.device)

    def is_ready(self) -> bool:
        """
        Check if the scorer is ready for use.

        Returns:
            True if the model and preprocessing are loaded and ready
        """
        return self._model is not None and self._preprocess is not None

    def __repr__(self) -> str:
        """String representation."""
        return f"CLIPScorer(model={self.clip_model_name}, device={self.device}, ready={self.is_ready()})"
