import logging
from typing import Dict, List, Optional, Any
from PIL import Image

from .base import BaseScorer
from .config import ScoringConfig
from .clip import CLIPScorer
from .aesthetic import AestheticScorer
from .hpsv2 import HPSv2Scorer

logger = logging.getLogger(__name__)


class Scorer:
    """
    Manager for coordinating multiple scoring methods.

    This class provides a unified interface for using multiple scorers
    and managing their configurations.
    """

    # Mapping of scorer names to their classes
    SCORER_CLASSES = {
        "clip": CLIPScorer,
        "aesthetic": AestheticScorer,
        "hpsv2": HPSv2Scorer,
    }

    def __init__(self, config: ScoringConfig):
        """
        Initialize the scoring manager.

        Args:
            config: Configuration for all scoring methods
        """
        self.config = config
        self.scorers: Dict[str, BaseScorer] = {}

        # Initialize enabled scorers
        self._initialize_scorers()

        logger.info(
            f"Scoring manager initialized with scorers: {self.get_active_methods()}"
        )

    def _initialize_scorers(self) -> None:
        """Initialize enabled scorers based on configuration."""
        for name, scorer_config in self.config.scorers.items():
            if scorer_config.enabled:
                scorer_class = self.SCORER_CLASSES.get(name)
                if scorer_class:
                    self.scorers[name] = scorer_class(scorer_config.config)

    def get_active_methods(self) -> List[str]:
        """Get list of active scoring method names."""
        return list(self.scorers.keys())

    def get_enabled_scorers(self) -> List[str]:
        """Get list of enabled scorer names (alias for get_active_methods)."""
        return self.get_active_methods()

    def get_scorer(self, name: str) -> Optional[BaseScorer]:
        """Get a specific scorer by name."""
        return self.scorers.get(name)

    def __call__(self, image: Image.Image, prompt: str) -> Dict[str, float]:
        """
        Score an image using all active scoring methods.

        Args:
            image: The image to score
            prompt: The text prompt to compare against

        Returns:
            Dictionary mapping scorer names to scores

        Raises:
            RuntimeError: If scoring fails for any method
        """
        scores = {}
        errors = []

        for name, scorer in self.scorers.items():
            try:
                score = scorer.score(image, prompt)
                scores[name] = score
                logger.debug(f"{name} score: {score:.4f}")
            except Exception as e:
                error_msg = f"{name} scoring failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            raise RuntimeError(
                f"Some scoring methods failed: {'; '.join(errors)}"
            )

        return scores

    def get_score_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active scoring methods."""
        info = {}
        for name, scorer in self.scorers.items():
            if hasattr(scorer, "get_score_info"):
                info[name] = scorer.get_score_info()
            else:
                info[name] = {
                    "name": name,
                    "class": scorer.__class__.__name__,
                    "ready": scorer.is_ready(),
                }
        return info

    def preload_models(self) -> None:
        """Preload all scoring models for faster execution."""
        logger.info("Preloading scoring models...")

        for name, scorer in self.scorers.items():
            try:
                if not scorer.is_ready():
                    logger.info(f"Preloading {name} model...")
                    scorer.load_model()
                    logger.info(f"✓ {name} model loaded")
                else:
                    logger.info(f"✓ {name} model already loaded")
            except Exception as e:
                logger.warning(f"Failed to preload {name} model: {str(e)}")

        logger.info("Scoring model preloading complete")

    def set_device(self, device: str) -> None:
        """Set device for all scorers."""
        for name, scorer in self.scorers.items():
            try:
                scorer.set_device(device)
                logger.debug(f"Set {name} scorer to device: {device}")
            except Exception as e:
                logger.warning(f"Failed to set device for {name}: {str(e)}")

    def __repr__(self) -> str:
        """String representation."""
        active_methods = self.get_active_methods()
        return f"ScoringManager(active_methods={active_methods}, config={self.config})"

    def _combine_scores(self, individual_scores: Dict[str, float]) -> float:
        """
        Combine individual scores using sophisticated weighting and conditional logic.

        This method implements a more sophisticated combining strategy when both CLIP and aesthetic
        scores are present, using weighted combinations and conditional logic to better balance
        prompt-image alignment with aesthetic quality.

        Args:
            individual_scores: Dictionary mapping scorer names to scores

        Returns:
            Combined score
        """
        # If we only have one score, return it directly
        if len(individual_scores) == 1:
            return list(individual_scores.values())[0]

        # Check if we have both CLIP and aesthetic scores
        has_clip = "clip" in individual_scores
        has_aesthetic = "aesthetic" in individual_scores
        # Score combining configuration
        # Get configuration values
        alpha = 1
        beta = 0.5
        gamma0 = 0.25
        clip_gate = 0.28
        clip_penalty_scale = 20.0
        clip_penalty_offset = 5.6

        if has_clip and has_aesthetic:
            # Sophisticated combining method for CLIP + Aesthetic
            clip_score = individual_scores["clip"]
            aesthetic_score = individual_scores["aesthetic"]
            clip_score = clip_score / 100
            # CLIP penalty: linearly map and clamp to <= 0, so only penalizes low match
            # This creates a penalty for poor prompt-image alignment
            r_clip_pen = min(
                clip_score * clip_penalty_scale - clip_penalty_offset, 0.0
            )

            # Gate aesthetics by CLIP match - only consider aesthetic score if CLIP alignment is good
            gamma = gamma0 if clip_score > clip_gate else 0.0

            # Combine scores with other scorers (HPS, etc.)
            other_scores = {
                k: v
                for k, v in individual_scores.items()
                if k not in ["clip", "aesthetic"]
            }
            other_score_sum = (
                sum(other_scores.values()) if other_scores else 0.0
            )

            combined = (
                alpha * other_score_sum
                + beta * r_clip_pen
                + gamma * aesthetic_score
            )

            logger.debug(
                f"Combined scores - CLIP: {clip_score:.4f}, Aesthetic: {aesthetic_score:.4f}, "
                f"Other: {other_score_sum:.4f}, Combined: {combined:.4f}"
            )

            return combined

        elif has_clip and not has_aesthetic:
            # Only CLIP score - apply penalty for poor alignment
            clip_score = individual_scores["clip"]
            clip_penalty = min(
                clip_penalty_scale * clip_score - clip_penalty_offset, 0.0
            )

            other_scores = {
                k: v for k, v in individual_scores.items() if k != "clip"
            }
            other_score_sum = (
                sum(other_scores.values()) if other_scores else 0.0
            )

            return other_score_sum + beta * clip_penalty

        elif has_aesthetic and not has_clip:
            # Only aesthetic score - use with reduced weight since no CLIP validation
            aesthetic_score = individual_scores["aesthetic"]
            other_scores = {
                k: v for k, v in individual_scores.items() if k != "aesthetic"
            }
            other_score_sum = (
                sum(other_scores.values()) if other_scores else 0.0
            )

            return (
                other_score_sum + 0.1 * aesthetic_score
            )  # Reduced weight without CLIP validation

        else:
            # Neither CLIP nor aesthetic - fall back to simple sum
            return sum(individual_scores.values())
