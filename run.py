import argparse
import json
import os
import logging
import yaml
from pathlib import Path
from typing import Iterable, Dict, Any
from tqdm import tqdm
from itertools import islice

from src.scoring import create_scoring_config
from src.pipeline import Pipeline
from src.utils import read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def run_batch(
    prompts_file: Path,
    output_root: Path,
    iterations: int,
    scoring_methods: Iterable[str],
    llm_engine: str,
    device: str,
    save_intermediate: bool,
    model_config: str,
    start_index: int,
    limit: int,
) -> None:
    # Create scoring configuration
    scoring_config = create_scoring_config(
        scoring_methods=scoring_methods, device=device
    )
    # Initialize pipeline
    pipeline = Pipeline(
        llm_engine=llm_engine,
        scoring_config=scoring_config,
        model_config=model_config,
    )

    prompts_iter = read_jsonl(prompts_file)

    # Apply start_index and limit slicing
    if limit is not None:
        prompts_iter = islice(prompts_iter, limit)

    for idx, row in enumerate(tqdm(prompts_iter, desc="Processing prompts")):
        if idx < start_index:
            continue

        prompt = (
            row.get("name")
            or row.get("initial_prompt")
            or row.get("prompt")
            or row.get("prompts")["base"][0]
            or ""
        )
        if not prompt:
            continue
        subdir_name = f"{idx:04d}"
        out_dir = output_root / subdir_name

        # Create output directory
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run optimization
        results = pipeline.run(
            prompt=prompt,
            iterations=iterations,
            save_intermediate=save_intermediate,
            output_dir=str(output_path),
        )

        # Save final results
        results_file = output_path / "optimization_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run image generation with optimization"
    )

    # Base config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_base.yaml",
        help="Path to base YAML config; CLI flags override values if provided",
    )

    # Optional overrides (default None so they don't override unless provided)
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to JSONL with prompts",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root output directory for all runs",
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Iterations per prompt"
    )
    parser.add_argument(
        "--scoring",
        type=str,
        nargs="+",
        default=None,
        help="Scoring methods (space-separated)",
    )
    parser.add_argument(
        "--llm_engine", 
        type=str, 
        default="azure/gpt-4o", 
        help="LLM engine"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Device for generation/scoring"
    )
    parser.add_argument(
        "--save_intermediate",
        type=lambda v: str(v).lower() in ("1", "true", "yes"),
        default=None,
        help="Save intermediate results (true/false)",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Model configuration",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start from this prompt index",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of prompts to process",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base config
    config_path = Path(os.path.abspath(args.config))
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")
    with config_path.open("r") as f:
        base_cfg: Dict[str, Any] = yaml.safe_load(f)

    def override(cfg: Dict[str, Any], key: str, value: Any) -> None:
        if value is not None:
            cfg[key] = value

    override(base_cfg, "prompts_file", args.prompts_file)
    override(base_cfg, "output_root", args.output_root)
    override(base_cfg, "iterations", args.iterations)
    override(base_cfg, "scoring", args.scoring)
    override(base_cfg, "llm_engine", args.llm_engine)
    override(base_cfg, "device", args.device)
    override(base_cfg, "save_intermediate", args.save_intermediate)
    override(base_cfg, "model_config", args.model_config)
    override(base_cfg, "start_index", args.start_index)
    override(base_cfg, "limit", args.limit)
    # Resolve paths
    prompts_file = Path(os.path.abspath(base_cfg["prompts_file"]))
    output_root = Path(os.path.abspath(base_cfg["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)

    run_batch(
        prompts_file=prompts_file,
        output_root=output_root,
        iterations=int(base_cfg["iterations"]),
        scoring_methods=base_cfg["scoring"],
        llm_engine=str(base_cfg["llm_engine"]),
        device=str(base_cfg["device"]),
        save_intermediate=bool(base_cfg["save_intermediate"]),
        model_config=str(base_cfg["model_config"]),
        start_index=int(base_cfg["start_index"]),
        limit=int(base_cfg["limit"]) if base_cfg["limit"] is not None else None,
    )


if __name__ == "__main__":
    main()
