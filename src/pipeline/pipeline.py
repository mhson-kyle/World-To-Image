import logging
import time
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from diffusers.utils import load_image
from dataclasses import dataclass, field
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection
from functools import partial
import torch
from PIL import Image
import os
import requests

from ..optim import ImageRetriever, Orchestrator, PromptOptimizer
from ..scoring import Scorer
from ..scoring import ScoringConfig
import sys

from typing import Tuple
from PIL import ImageOps
import matplotlib.pyplot as plt

import accelerate
from torchvision.transforms.functional import to_tensor


root_dir = Path().resolve()
sys.path.append(str(root_dir))
omnigen2_dir = root_dir / "OmniGen2"
sys.path.append(str(omnigen2_dir))

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import (
    OmniGen2Transformer2DModel,
)
from omnigen2.utils.img_util import create_collage


logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Session-level metadata for optimization runs."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    original_prompt: str = ""
    total_iterations: int = 0
    best_iteration: int = 0
    best_score: float = float("-inf")
    optimization_strategy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "original_prompt": self.original_prompt,
            "total_iterations": self.total_iterations,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "optimization_strategy": self.optimization_strategy,
        }


@dataclass
class ReferenceImageInfo:
    """Information about reference images used in generation."""

    source: str  # "ip_adapter", "user_provided", etc.
    query: str
    image_path: str
    scale: float = 0.0
    category: str = "general"  # "content", "style", "general"
    retrieval_confidence: float = 0.0
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "query": self.query,
            "image_path": self.image_path,
            "scale": self.scale,
            "category": self.category,
            "retrieval_confidence": self.retrieval_confidence,
            "index": self.index,
        }


@dataclass
class GenerationResult:
    """Result of enhanced optimization with hierarchical metadata."""

    # Basic optimization info
    iteration: int
    prompt: str
    generated_image: Image.Image
    # Scores
    individual_scores: Dict[str, float]
    combined_score: float

    # Optional fields (with defaults) must come after non-default fields
    image_retrieval_result: Optional[str] = None
    prompt_optimizer_result: Optional[Dict[str, Any]] = None

    # Optimization details
    enhancements: Optional[str] = None
    visual_analysis: Optional[str] = None

    # Prompt evolution tracking
    prompt_evolution: Optional[Dict[str, Any]] = None

    # Orchestrator decision tracking
    orchestrator_decision: Optional[Dict[str, Any]] = None

    # Reference images with detailed info
    reference_images: List[ReferenceImageInfo] = field(default_factory=list)

    # Timing metadata
    generation_time: float = 0.0
    scoring_time: float = 0.0
    optimization_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to hierarchical dictionary structure."""
        return {
            "iteration": self.iteration,
            "prompt_evolution": {
                "original": self.prompt_evolution.get("original", "")
                if self.prompt_evolution
                else "",
                "optimized": self.prompt,
                "changes": self.prompt_evolution.get("changes", [])
                if self.prompt_evolution
                else [],
                "optimization_reasoning": self.enhancements or "",
            },
            "orchestrator_decision": self.orchestrator_decision or {},
            "prompt_optimizer_result": self.prompt_optimizer_result or {},
            "scores": {
                "individual_scores": self.individual_scores,
                "combined_score": self.combined_score,
            },
            "visual_analysis": self.visual_analysis,
            "image_retrieval_result": self.image_retrieval_result,
            "reference_images": [
                ref.to_dict() for ref in self.reference_images
            ],
            "image_metadata": {
                "size": f"{self.generated_image.size[0]}x{self.generated_image.size[1]}",
                "format": self.generated_image.format or "PNG",
            },
            "timing": {
                "generation_time": self.generation_time,
                "scoring_time": self.scoring_time,
                "optimization_time": self.optimization_time,
                "total_time": self.generation_time
                + self.scoring_time
                + self.optimization_time,
            },
        }


class Pipeline:
    """
    Pipeline that integrates model selection, hyperparameter optimization,
    and generation capabilities.
    """

    def __init__(
        self,
        llm_engine: str,
        scoring_config: ScoringConfig,
        model_config: str,
    ):
        """
        Initialize the enhanced pipeline.

        Args:
            llm_engine: LLM engine
            scoring_config: Scoring configuration
            model_config: Model configuration
        """
        # Initialize orchestrator
        self.llm_engine = llm_engine
        self.orchestrator = Orchestrator(llm_engine=self.llm_engine)
        self.scorer = Scorer(scoring_config)

        # with open(model_config, "r") as f:
        #     self.model_config = json.load(f)
        # self.parser = PromptParser(self.llm_engine)

        # Initialize OmniGen2 pipeline
        self.accelerator = accelerate.Accelerator()
        model_path = "OmniGen2/OmniGen2"
        self.pipe = OmniGen2Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token="",
        )
        self.pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.pipe = self.pipe.to(self.accelerator.device, dtype=torch.bfloat16)

        # Pipeline state
        # self.current_model_info = "" # self._create_default_model_info()
        # self.current_hyperparameters = "" # self._get_default_hyperparameters()
        self.optimization_history: List[GenerationResult] = []

        # Session metadata
        self.session_metadata: Optional[SessionMetadata] = None

        logger.info("Pipeline initialized successfully")

    # def _get_default_hyperparameters(self) -> Dict[str, Any]:
    #     """Get default hyperparameters from model config."""
    #     return {
    #         "guidance_scale": self.model_config["pipeline_call_args"].get("guidance_scale", 7.5),
    #         "num_inference_steps": self.model_config["pipeline_call_args"].get("num_inference_steps", 50),
    #         "height": self.model_config["pipeline_call_args"].get("height", 1024),
    #         "width": self.model_config["pipeline_call_args"].get("width", 1024),
    #         "negative_prompt": self.model_config["pipeline_call_args"].get("negative_prompt", ""),
    #     }

    # def _create_default_model_info(self) -> ModelInfo:
    #     """Create default model info from config."""
    #     return ModelInfo(
    #         id=self.model_config["pretrained_model_name_or_path"],
    #         name=self.model_config["pretrained_model_name_or_path"],
    #         model_type="diffusers",
    #         base_model=self.model_config["pretrained_model_name_or_path"],
    #         default_hyperparameters=self._get_default_hyperparameters(),
    #     )

    def run(
        self,
        prompt: str,
        iterations: int = 5,
        save_intermediate: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the enhanced optimization pipeline.

        Args:
            prompt: Input prompt to optimize
            iterations: Number of optimization iterations
            save_intermediate: Whether to save intermediate results
            output_dir: Output directory for results

        Returns:
            Dictionary containing optimization results and metadata
        """
        self.original_prompt = prompt
        self.current_prompt = prompt
        self.negative_prompt = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
        self.enhancements = ""
        self.visual_analysis = ""
        self.missing_keywords = []

        # Initialize session metadata
        self.session_metadata = SessionMetadata(
            original_prompt=self.original_prompt,
            total_iterations=iterations,
            optimization_strategy="prompt_and_image_retrieval",  # Will be updated based on actual strategies used
        )

        all_images = []
        self.reference_image = all_images if all_images else None
        # Initialize best result tracking
        self.best_result = None
        self.best_combined_score = float("-inf")

        # Track strategies used across iterations
        strategies_used = set()

        for iteration in range(iterations):
            logger.info(f"Iteration {iteration + 1}/{iterations}")

            ######### Step 1: Generate image with current prompt and hyperparameters #########
            start_time = time.time()
            self.generated_image = self._generate_image(
                prompt=self.current_prompt,
                input_images=self.reference_image,
                negative_prompt=self.negative_prompt,
            )
            generation_time = time.time() - start_time

            ######### Step 2: Score the generated image ##########
            start_time = time.time()
            self.individual_scores = self.scorer(
                image=self.generated_image,
                prompt=self.current_prompt,
            )
            self.combined_score = self.scorer._combine_scores(
                individual_scores=self.individual_scores,
            )
            scoring_time = time.time() - start_time
            #######################################################

            ######### Step 3: Log the result ##########
            result = GenerationResult(
                iteration=iteration + 1,
                prompt=self.current_prompt,
                generated_image=self.generated_image,
                individual_scores=self.individual_scores,
                combined_score=self.combined_score,
                enhancements=self.enhancements,
                visual_analysis=self.visual_analysis,
                image_retrieval_result="",  # self.image_retrieval_result['reasoning'],
                prompt_evolution={
                    "original": self.original_prompt
                    if iteration == 0
                    else self.optimization_history[-1].prompt,
                    "changes": [],  # Will be populated by orchestrator
                },
                generation_time=generation_time,
                scoring_time=scoring_time,
            )

            if (
                self.best_result is None
                or self.combined_score > self.best_combined_score
            ):
                self.best_combined_score = self.combined_score
                self.best_result = result

            self.optimization_history.append(result)

            if save_intermediate and output_dir:
                self._save_intermediate_result(
                    result=result,
                    iteration=iteration,
                    output_dir=output_dir,
                    reference_image=self.reference_image,
                )
            #######################################################

            ######### Orchestrator-based optimization ##########
            if iteration < iterations - 1:
                ######### Step 1: Use orchestrator to analyze and optimize ##########
                start_time = time.time()
                orchestrator_result = self.orchestrator(
                    current_prompt=self.current_prompt,
                    image=self.generated_image,
                    current_scores=self.individual_scores,
                    original_prompt=self.original_prompt,  # Pass original prompt
                    iteration=iteration,
                    optimization_history=[
                        result.to_dict() for result in self.optimization_history
                    ],
                )
                optimization_time = time.time() - start_time
                result.optimization_time = optimization_time

                # Capture orchestrator decision metadata
                result.orchestrator_decision = {
                    "task_type": orchestrator_result.get("task_type", ""),
                    "reasoning": orchestrator_result.get("reasoning", ""),
                    "references_needed": orchestrator_result.get(
                        "references_needed", []
                    ),
                    "final_prompt": orchestrator_result.get("final_prompt", ""),
                    "confidence": orchestrator_result.get("confidence", 0.0),
                }
                # Track strategies used (derive from task_type and references_needed)
                if orchestrator_result.get("references_needed"):
                    strategies_used.update(
                        ["prompt_optimizer", "image_retrieval"]
                    )
                else:
                    strategies_used.update(["prompt_optimizer"])

                ######### Apply optimization results ##########
                ######## Image editing with prompt ##########
                if (
                    orchestrator_result["task_type"]
                    == "image_editing_with_prompt"
                ):
                    self.reference_image = [self.generated_image]
                    result.image_retrieval_result = orchestrator_result[
                        "optimization_results"
                    ]["image_retrieval"]["reasoning"]
                ######## Image editing with prompt and reference ##########
                elif (
                    orchestrator_result["task_type"]
                    == "image_editing_with_prompt_and_reference"
                ):
                    self.reference_image = [self.generated_image]
                    image_retrieval_result = orchestrator_result[
                        "optimization_results"
                    ]["image_retrieval"]
                    retrieved_images = (
                        self._extract_images_from_retrieval_result(
                            image_retrieval_result
                        )
                    )
                    self.reference_image.extend(retrieved_images)
                    result.image_retrieval_result = image_retrieval_result[
                        "reasoning"
                    ]
                ######## Text to image ##########
                else:
                    # Handle additional optimization results if available
                    if (
                        "optimization_results" in orchestrator_result
                        and "prompt_optimizer"
                        in orchestrator_result["optimization_results"]
                    ):
                        prompt_result = orchestrator_result[
                            "optimization_results"
                        ]["prompt_optimizer"]
                        result.prompt_optimizer_result = prompt_result
                        # Extract negative prompts for next generation
                        if "negative_prompts" in prompt_result:
                            self.negative_prompt = prompt_result[
                                "negative_prompts"
                            ]
                            logger.info(
                                f"Updated negative prompt: {self.negative_prompt}"
                            )
                        self.visual_analysis = prompt_result.get(
                            "visual_analysis", ""
                        )

                    ######## Image retrieval results for text_image_to_image task ##########
                    if (
                        "optimization_results" in orchestrator_result
                        and "image_retrieval"
                        in orchestrator_result["optimization_results"]
                    ):
                        image_retrieval_result = orchestrator_result[
                            "optimization_results"
                        ]["image_retrieval"]
                        result.image_retrieval_result = image_retrieval_result[
                            "reasoning"
                        ]
                        retrieved_images = (
                            self._extract_images_from_retrieval_result(
                                image_retrieval_result
                            )
                        )

                        if (
                            orchestrator_result["task_type"]
                            == "text_image_to_image"
                        ):
                            self.reference_image = retrieved_images
                            logger.info(
                                f"Updated reference images: {len(self.reference_image)} images from image retrieval"
                            )
                            logger.info(
                                f"Image retrieval reasoning: {image_retrieval_result.get('reasoning', '')}"
                            )
                            result.image_retrieval_result = (
                                image_retrieval_result["reasoning"]
                            )
                        # Create reference image metadata for next iteration
                        # Note: This will be applied to the next iteration's result
                        next_reference_info = []
                        if retrieved_images:
                            queries = image_retrieval_result.get("queries", [])
                            for idx, ref_img in enumerate(retrieved_images):
                                ref_info = ReferenceImageInfo(
                                    source="image_retrieval",
                                    query=queries[idx]
                                    if idx < len(queries)
                                    else "",
                                    image_path=f"reference/iteration_{iteration + 2:02d}_reference_{idx}.png",
                                    scale=1.0,  # Default scale for image retrieval
                                    category="general",
                                    retrieval_confidence=0.9,  # Default confidence
                                    index=idx,
                                )
                                next_reference_info.append(ref_info)
                ######## Use the final prompt from orchestrator ##########
                if orchestrator_result.get("final_prompt"):
                    self.current_prompt = orchestrator_result["final_prompt"]
                    logger.info(f"Updated prompt: {self.current_prompt}")
            logger.info(
                f"Iteration {iteration + 1} completed. Score: {self.combined_score:.4f}"
            )
            ###############################################################
        logger.info(
            f"Optimization completed. Best score: {self.best_combined_score:.4f}"
        )

        ######## End of optimization ##########
        if self.session_metadata:
            self.session_metadata.best_iteration = self.best_result.iteration
            self.session_metadata.best_score = self.best_result.combined_score
            self.session_metadata.optimization_strategy = (
                ", ".join(sorted(strategies_used))
                if strategies_used
                else "none"
            )

        self._save_best_result(
            self.best_result, output_dir, self.reference_image
        )

        final_results = self._generate_final_results()
        self.reset()
        ######################################################
        return final_results

    def _generate_image(
        self,
        prompt: str,
        input_images: Optional[List[Image.Image]] = None,
        negative_prompt: str = "",
    ) -> Image.Image:
        """Generate image with current hyperparameters."""
        # Preprocess input images if provided
        if input_images:
            # Handle different types of input images
            if isinstance(input_images, list) and len(input_images) > 0:
                # Check if input_images are PIL Images (from IP-adapter) or paths (from preprocess)
                if isinstance(input_images[0], Image.Image):
                    # Already PIL Images from IP-adapter, no preprocessing needed
                    pass
                elif isinstance(input_images[0], str):
                    # File paths, need to preprocess
                    input_images = preprocess(input_images)
            elif isinstance(input_images, str):
                # Single path
                input_images = preprocess([input_images])
        else:
            input_images = []
        logging.info(
            f"-------------- {len(input_images)} input images --------------"
        )
        # Generate with dynamic seed based on iteration for variety
        generator = torch.Generator(device=self.accelerator.device).manual_seed(
            0
        )

        results = self.pipe(
            prompt=prompt,
            input_images=input_images,
            width=1024,
            height=1024,
            num_inference_steps=50,
            max_sequence_length=1024,
            text_guidance_scale=5.0,
            image_guidance_scale=2.0,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        return results.images[0]

    def _save_intermediate_result(
        self,
        result: GenerationResult,
        iteration: int,
        output_dir: str,
        reference_image: Optional[List[Image.Image]] = None,
    ):
        """Save intermediate result to disk with hierarchical metadata structure."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save generated image
        image_path = output_path / f"iteration_{iteration + 1:02d}.png"
        result.generated_image.save(image_path)

        # Save reference images in a subfolder if they exist
        if reference_image and len(reference_image) > 0:
            # Create reference subfolder
            reference_dir = output_path / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)

            # Update reference image paths in result metadata
            for idx, ref_img in enumerate(reference_image):
                if isinstance(ref_img, Image.Image):
                    ref_img_path = (
                        reference_dir
                        / f"iteration_{iteration + 1:02d}_reference_{idx}.png"
                    )
                    ref_img.save(ref_img_path)

                    # Update the reference image info in the result
                    if idx < len(result.reference_images):
                        result.reference_images[idx].image_path = str(
                            ref_img_path.relative_to(output_path)
                        )
                    else:
                        # Create new reference image info if not exists
                        ref_info = ReferenceImageInfo(
                            source="image_retrieval",
                            query="",
                            image_path=str(
                                ref_img_path.relative_to(output_path)
                            ),
                            scale=1.0,
                            category="general",
                            retrieval_confidence=0.9,
                            index=idx,
                        )
                        result.reference_images.append(ref_info)

        # Save session metadata (only once, on first iteration)
        if iteration == 0 and self.session_metadata:
            session_metadata_path = output_path / "session_metadata.json"
            with open(session_metadata_path, "w") as f:
                json.dump(
                    self.session_metadata.to_dict(), f, indent=2, default=str
                )

        # Save iteration metadata with hierarchical structure
        metadata_path = (
            output_path / f"iteration_{iteration + 1:02d}_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(
            f"Saved iteration {iteration + 1} with {len(result.reference_images)} reference images"
        )

    def _save_best_result(
        self,
        result: GenerationResult,
        output_dir: str,
        reference_image: Optional[List[Image.Image]] = None,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save best result image
        image_path = output_path / "best_result.png"
        result.generated_image.save(image_path)

        # Save reference images for best result if they exist
        if reference_image and len(reference_image) > 0:
            # Create reference subfolder
            reference_dir = output_path / "reference"
            reference_dir.mkdir(parents=True, exist_ok=True)

            for idx, ref_img in enumerate(reference_image):
                if isinstance(ref_img, Image.Image):
                    ref_img_path = (
                        reference_dir / f"best_result_reference_{idx}.png"
                    )
                    ref_img.save(ref_img_path)

            logger.info(
                f"Saved best result with {len(reference_image)} reference images"
            )

        # Save best result metadata
        best_result_metadata_path = output_path / "best_result_metadata.json"
        with open(best_result_metadata_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Update and save final session metadata
        if self.session_metadata:
            session_metadata_path = output_path / "session_metadata.json"
            with open(session_metadata_path, "w") as f:
                json.dump(
                    self.session_metadata.to_dict(), f, indent=2, default=str
                )

    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate the final results summary with hierarchical metadata."""
        # Find best result
        best_result = max(
            self.optimization_history, key=lambda x: x.combined_score
        )

        # Calculate statistics
        all_scores = [
            result.combined_score for result in self.optimization_history
        ]
        score_improvements = []
        for i in range(1, len(all_scores)):
            improvement = all_scores[i] - all_scores[i - 1]
            score_improvements.append(improvement)

        # Create hierarchical final results
        final_results = {
            "session": self.session_metadata.to_dict()
            if self.session_metadata
            else {},
            "optimization_summary": {
                "total_iterations": len(self.optimization_history),
                "best_iteration": best_result.iteration,
                "best_combined_score": best_result.combined_score,
                "best_individual_scores": best_result.individual_scores,
                "score_improvements": score_improvements,
                "final_prompt": best_result.prompt,
                "convergence_analysis": {
                    "score_trajectory": all_scores,
                    "total_improvement": all_scores[-1] - all_scores[0]
                    if len(all_scores) > 1
                    else 0,
                    "average_improvement_per_iteration": sum(score_improvements)
                    / len(score_improvements)
                    if score_improvements
                    else 0,
                },
            },
            "optimization_analytics": {
                "strategies_used": list(
                    set(
                        strategy
                        for result in self.optimization_history
                        if result.orchestrator_decision
                        for strategy in result.orchestrator_decision.get(
                            "strategies", []
                        )
                    )
                ),
                "reference_images_used": sum(
                    len(result.reference_images)
                    for result in self.optimization_history
                ),
                "total_optimization_time": sum(
                    result.optimization_time
                    for result in self.optimization_history
                ),
                "total_generation_time": sum(
                    result.generation_time
                    for result in self.optimization_history
                ),
                "total_scoring_time": sum(
                    result.scoring_time for result in self.optimization_history
                ),
            },
            "scoring_methods": self.scorer.get_active_methods(),
            "iterations": [
                result.to_dict() for result in self.optimization_history
            ],
        }

        logger.info("Pipeline completed")
        return final_results

    def _extract_images_from_retrieval_result(
        self, image_retrieval_result: Dict[str, Any]
    ) -> List[Image.Image]:
        """Extract PIL Images from image retrieval result.

        The IP adapter returns images in format: {"images": [...], "queries": [...], "reasoning": "..."}
        where images is a list of lists - one list per query, each containing tuples of (image, score).
        The image is a tuple of (image, score).
        """
        retrieved_images = []
        images = image_retrieval_result["images"]
        for image in images:
            retrieved_images.append(image[0])
        logger.info(
            f"Extracted {len(retrieved_images)} images from retrieval result"
        )
        return retrieved_images

    def reset(self) -> None:
        """Reset the pipeline state."""
        self.optimization_history.clear()
        self.session_metadata = None
        logger.info("Pipeline reset")


def preprocess(
    input_image_path: List[str] = [],
) -> Tuple[str, str, List[Image.Image]]:
    """Preprocess the input images."""
    # Process input images
    input_images = []

    if input_image_path:
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [
                Image.open(os.path.join(input_image_path[0], f))
                for f in os.listdir(input_image_path[0])
            ]
        else:
            input_images = [Image.open(path) for path in input_image_path]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images
