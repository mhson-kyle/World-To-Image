from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any
from PIL import Image
import torch


class BaseScorer(ABC):
    """
    Abstract base class for scoring mechanisms.

    This class defines the interface that all scoring mechanisms must implement,
    allowing for consistent evaluation of image-prompt alignment and generation quality.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the scorer with optional configuration.

        Args:
            config: Configuration dictionary for the scorer
        """
        self.config = config or {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = None

    @abstractmethod
    def score(self, image: Image.Image, prompt: str, **kwargs) -> float:
        """
        Score the alignment between an image and a prompt.

        Args:
            image: The image to score
            prompt: The text prompt to compare against
            **kwargs: Additional scoring parameters

        Returns:
            Score between 0 and 1, where higher values indicate better alignment

        Raises:
            NotImplementedError: If the scorer doesn't implement this method
            RuntimeError: If scoring fails
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the underlying scoring model.

        This method should load the model into memory and prepare it
        for scoring. It may be called automatically during initialization
        or explicitly by the user.

        Raises:
            RuntimeError: If model loading fails
        """
        pass

    def set_device(self, device: Union[str, torch.device]) -> None:
        """
        Set the device for model execution.

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Move model to device if it's loaded
        if self._model is not None:
            self._model = self._model.to(device)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def update_config(self, **kwargs) -> None:
        """
        Update the configuration with new parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)

    def is_ready(self) -> bool:
        """
        Check if the scorer is ready for use.

        Returns:
            True if the model is loaded and ready
        """
        return self._model is not None

    def __enter__(self):
        """Context manager entry."""
        if not self.is_ready():
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Clean up resources if needed
        if hasattr(self, "_cleanup"):
            self._cleanup()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(device={self.device}, ready={self.is_ready()})"
