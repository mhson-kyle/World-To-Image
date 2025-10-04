from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """Configuration for a single scorer."""

    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.weight < 0:
            raise ValueError(
                f"Scorer weight must be non-negative, got {self.weight}"
            )


@dataclass
class ScoringConfig:
    """Configuration for all scoring methods."""

    # Dynamic scorer configurations - automatically populated
    scorers: Dict[str, ScorerConfig] = field(default_factory=dict)

    # Global scoring settings
    default_device: str = "cuda"
    cache_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set default device for all scorers
        for scorer_config in self.scorers.values():
            if scorer_config.enabled and "device" not in scorer_config.config:
                scorer_config.config["device"] = self.default_device

            # Set cache directory if specified
            if (
                scorer_config.enabled
                and self.cache_dir
                and "cache_dir" not in scorer_config.config
            ):
                scorer_config.config["cache_dir"] = self.cache_dir

    def validate(self):
        """Validate that at least one scorer is enabled."""
        enabled_scorers = [
            name for name, config in self.scorers.items() if config.enabled
        ]
        if not enabled_scorers:
            raise ValueError("At least one scorer must be enabled")

    def get_enabled_scorers(self) -> List[str]:
        """Get list of enabled scorer names."""
        return [name for name, config in self.scorers.items() if config.enabled]

    def get_scorer_config(self, name: str) -> Optional[ScorerConfig]:
        """Get configuration for a specific scorer."""
        return self.scorers.get(name)

    def add_scorer(
        self,
        name: str,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ):
        """Add or update a scorer configuration."""
        self.scorers[name] = ScorerConfig(
            name=name, enabled=enabled, config=config or {}, weight=weight
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "scorers": {
                name: {
                    "enabled": config.enabled,
                    "config": config.config,
                    "weight": config.weight,
                }
                for name, config in self.scorers.items()
            },
            "default_device": self.default_device,
            "cache_dir": self.cache_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ScoringConfig":
        """Create configuration from dictionary."""
        config = cls(
            default_device=config_dict.get("default_device", "cuda"),
            cache_dir=config_dict.get("cache_dir"),
        )

        # Add scorers from dictionary
        for name, scorer_dict in config_dict.get("scorers", {}).items():
            config.add_scorer(
                name=name,
                enabled=scorer_dict.get("enabled", True),
                config=scorer_dict.get("config", {}),
                weight=scorer_dict.get("weight", 1.0),
            )

        # Validate after all scorers are added
        config.validate()
        return config

    @classmethod
    def create_default(
        cls, scoring_methods: Optional[List[str]] = None, device: str = "cuda"
    ) -> "ScoringConfig":
        """Create default configuration with specified scoring methods."""
        config = cls(default_device=device)

        # Default to HPSv2 only if no methods specified (more memory efficient)
        if scoring_methods is None:
            scoring_methods = ["hpsv2"]

        # Add each scoring method
        for method in scoring_methods:
            if method == "hpsv2":
                config.add_scorer("hpsv2", enabled=True, weight=1.0)
            elif method == "hpsv3":
                config.add_scorer("hpsv3", enabled=True, weight=1.0)
            elif method == "clip":
                config.add_scorer("clip", enabled=True, weight=1.0)
            elif method == "aesthetic":
                config.add_scorer("aesthetic", enabled=True, weight=1.0)
            else:
                # For any future scoring methods, just add them
                config.add_scorer(method, enabled=True, weight=1.0)

        # Validate after all scorers are added
        config.validate()
        return config

    def __repr__(self) -> str:
        """String representation."""
        enabled = self.get_enabled_scorers()
        return f"ScoringConfig(enabled={enabled}, device={self.default_device})"


def create_scoring_config(
    scoring_methods: Optional[List[str]] = None,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    **kwargs,
) -> ScoringConfig:
    """
    Create a scoring configuration with specified settings.

    Args:
        scoring_methods: List of scoring methods to enable
        device: Default device for scorers
        cache_dir: Directory for caching model weights
        **kwargs: Additional configuration options

    Returns:
        Configured ScoringConfig instance
    """
    config = ScoringConfig.create_default(scoring_methods, device)

    if cache_dir:
        config.cache_dir = cache_dir

    # Apply additional configurations
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def load_scoring_config_from_file(
    config_path: Union[str, Path],
) -> ScoringConfig:
    """
    Load scoring configuration from a JSON or YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        ScoringConfig instance
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix.lower() == ".json":
        import json

        with open(config_path, "r") as f:
            config_dict = json.load(f)
    elif config_path.suffix.lower() in [".yaml", ".yml"]:
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported configuration file format: {config_path.suffix}"
        )

    return ScoringConfig.from_dict(config_dict)
