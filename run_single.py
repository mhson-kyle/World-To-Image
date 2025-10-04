import argparse
import json
import logging
from pathlib import Path

from src.pipeline import Pipeline
from src.scoring import create_scoring_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single prompt optimization"
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="Input prompt to optimize",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of optimization iterations (default: 5)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        nargs="+",
        default=["clip", "aesthetic"],
        help="Scoring methods to use (space-separated, e.g., clip aesthetic)",
    )
    parser.add_argument(
        "--llm_engine",
        type=str,
        default="azure/gpt-4o",
        help="LLM engine to use for optimization (default: azure/gpt-4o)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for generation and scoring (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        default=True,
        help="Save intermediate results (default: True)",
    )
    parser.add_argument(
        "--no_save_intermediate",
        dest="save_intermediate",
        action="store_false",
        help="Disable saving intermediate results",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="diffusion_configs/sdxl-base.json",
        help="Model configuration file (default: diffusion_configs/sdxl-base.json)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    # Create scoring configuration
    scoring_config = create_scoring_config(
        scoring_methods=args.scoring,
        device=args.device,
    )

    # Initialize pipeline
    pipeline = Pipeline(
        llm_engine=args.llm_engine,
        scoring_config=scoring_config,
        model_config=args.model_config,
    )

    # Run optimization
    results = pipeline.run(
        prompt=args.prompt,
        iterations=args.iterations,
        save_intermediate=args.save_intermediate,
        output_dir=str(output_path),
    )

    # Save final results
    results_file = output_path / "optimization_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Optimization completed. Results saved to {output_path}")


if __name__ == "__main__":
    main()


