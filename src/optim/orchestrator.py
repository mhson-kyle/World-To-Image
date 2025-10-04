import logging
import json
import base64
import io
from typing import Dict, List, Optional, Any, Union
from PIL import Image

from .base_llm_engine import BaseLLMEngine
from .prompt_optimizer import PromptOptimizer
from .image_retriever import ImageRetriever

logger = logging.getLogger(__name__)


class Orchestrator(BaseLLMEngine):
    """
    Orchestrator for a multimodal generation system that coordinates two collaborators:
    - Prompt Optimizer (PO): rewrites prompts, decides what reference images are needed
    - Image Retrieval (IR): finds reference images for each needed concept

    Decides between 4 task types and coordinates PO and IR as a team.
    """

    def __init__(self, llm_engine: str = "azure/gpt-4o"):
        """Initialize the orchestrator with optimization agents."""
        super().__init__(llm_engine)

        # Initialize optimization agents
        self.prompt_optimizer = PromptOptimizer(llm_engine=llm_engine)
        self.image_retrieval_optimizer = ImageRetriever(llm_engine=llm_engine)

    def __call__(
        self,
        current_prompt: str,
        image: Optional[Image.Image] = None,
        current_scores: Optional[Dict[str, float]] = None,
        original_prompt: Optional[str] = None,
        iteration: Optional[int] = None,
        optimization_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        logging.info("-------------- Orchestrator called --------------")

        # Create prompt
        visual_analysis = self._get_visual_analysis(
            original_prompt,
            image,
        )
        analysis_prompt = self._create_prompt(
            original_prompt=original_prompt,
            current_prompt=current_prompt,
            current_scores=current_scores,
            optimization_history=optimization_history,
            visual_analysis=visual_analysis,
            # keywords, style
        )

        # Prompt LLM
        response = self.engine(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are the Orchestrator for a multimodal generation system. You coordinate Prompt Optimizer (PO) and Image Retrieval (IR) to work together as a team.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_base64_url(image)
                            },
                        },
                    ],
                },
            ],
            temperature=0.1,
        )
        llm_response = response.choices[0].message.content

        # Parse response
        results = self._parse_response(llm_response)
        logging.info(
            f"-------------- Task type: {results['task_type']} --------------"
        )
        logging.info(
            f"Orchestrator decided on strategies: {results['strategies']} - {results['reasoning']}"
        )
        results["optimization_results"] = {}
        # Step 2: Coordinate PO and IR as a team
        if results["references_needed"]:
            # IR directly retrieves images based on orchestrator's decision
            logging.info("Step 2a: IR retrieving reference images...")
            ir_result = self.image_retrieval_optimizer(
                original_prompt=original_prompt or current_prompt,
                reasoning=results["reasoning"],
                queries=results[
                    "references_needed"
                ],  # Use orchestrator's decision directly
            )
            results["optimization_results"]["image_retrieval"] = ir_result

            # PO optimizes prompt knowing that reference images will be available
            logging.info(
                "Step 2b: PO optimizing prompt with reference context..."
            )
            po_result = self.prompt_optimizer(
                task_type=results["task_type"],
                current_prompt=current_prompt,
                original_prompt=original_prompt or current_prompt,
                image=image,
                current_scores=current_scores or {},
                iteration=iteration,
                reasoning=f"{results['reasoning']} - Reference images retrieved for: {results['references_needed']}",
            )
            results["final_prompt"] = po_result["optimized_prompt"]
            results["optimization_results"]["prompt_optimizer"] = po_result
        else:
            # No references needed, just optimize prompt for better image generation/editing
            logging.info(
                "Step 2: PO optimizing prompt for image improvement..."
            )
            po_result = self.prompt_optimizer(
                task_type=results["task_type"],
                current_prompt=current_prompt,  # Same prompt - improve the image
                original_prompt=original_prompt or current_prompt,
                image=image,
                current_scores=current_scores or {},
                iteration=iteration,
                reasoning=results["reasoning"],
            )

            results["final_prompt"] = po_result["optimized_prompt"]
            results["optimization_results"]["prompt_optimizer"] = po_result
            # No image retrieval was performed in this branch
            results["optimization_results"]["image_retrieval"] = {
                "images": [],
                "queries": [],
                "reasoning": "No reference images needed for this task type",
            }
        return results

    def _create_prompt(
        self,
        original_prompt: str,
        current_prompt: str,
        current_scores: Dict[str, float],
        optimization_history: Optional[List[Dict[str, Any]]] = None,
        visual_analysis: Optional[str] = None,
    ) -> str:
        """Create task classification and strategy selection prompt for OmniGen2 orchestration."""

        # keywords_str = f"KEYWORDS: {keywords}" if keywords else "KEYWORDS: Not provided"
        # style_str = f"STYLE ELEMENTS: {style}" if style else "STYLE ELEMENTS: Not provided"

        return f"""
        You are an expert orchestrator for the OmniGen2 multimodal generation model.
        Your job is to:
        1. Analyze the provided image, prompt, scores, and optimization history.
        2. Decide the most suitable generation task type: (This is in order of preference) 
            - **text_image_to_image**: Use a reference image + prompt for improved fidelity. (MOST RECOMMENDED)
            - **text_to_image**: Generate image purely from text prompt.
            - **image_editing_with_prompt_and_reference**: Modify the currently generated image according to the prompt and reference image.
            - **image_editing_with_prompt**: Modify the currently generated image according to the prompt (inpainting, style transfer, attribute edit).

        ## Guidelines
        - Image editing is the least recommended task type.
        - You should only choose image editing if the generated image is very good and you are confident that the prompt is not enough to improve the image.
        
        
        ## Inputs
        ORIGINAL PROMPT: "{original_prompt}"
        CURRENT OPTIMIZED PROMPT: "{current_prompt}"
        DETAILED_SCORES: {json.dumps(current_scores, indent=2)}
        OPTIMIZATION HISTORY: {json.dumps(optimization_history, indent=2)}
        VISUAL ANALYSIS: {visual_analysis}

        ## Task Classification Rules
        - **text_to_image**: Prompt is self-sufficient; no celebrity/IP likeness, no niche style, no need to preserve an existing image.
        - **text_image_to_image**: Prompt includes niche entities (celebrity/IP/meme), rare styles, or ambiguous visuals → retrieve TWO references.
        - **image_editing_with_prompt**: A previously generated image exists AND the new text indicates incremental change (style tweak, color, local edit) without needing a specific external reference.
        - **image_editing_with_prompt_and_reference**: A previously generated image exists AND the new text implies matching a specific look/scene/face/style from a known IP or example → retrieve ONE reference.

        ### Disambiguation (text-only prompts that might be edits)
        - If OPTIMIZATION_HISTORY shows a recent successful generation (e.g., within last step) and DETAILED_SCORES indicate high content alignment but style mismatch → prefer **image_editing_with_prompt**.
        - If the text asks to match a specific world/IP/location/face (e.g., “Shrek swamp”, “Monica’s apartment”, “Van Gogh brushwork”) → prefer **image_editing_with_prompt_and_reference**.
        - If structural changes are large (pose/layout/object count), or prior image is low-quality/incorrect content → prefer **text_image_to_image** (with references if niche) or **text_to_image**.
        - Reference needed should just be a simple keyword or a list of keywords.

        ## Strategy Selection
        - **text_to_image** → ["prompt_optimizer"]
        - **text_image_to_image** → ["prompt_optimizer", "image_retrieval"]
        - **image_editing_with_prompt** → ["prompt_optimizer"]
        - **image_editing_with_prompt_and_reference** → ["prompt_optimizer", "image_retrieval"]

        
        ## Output Format
        Return a JSON object:
        {{
            "task_type": "text_to_image" | "text_image_to_image" | "image_editing_with_prompt" | "image_editing_with_prompt_and_reference",
            "strategies": ["prompt_optimizer", "image_retrieval"],
            "references_needed": ["reference_image_1", "reference_image_2"],
            "draft_prompt": "Draft prompt for the prompt optimizer to optimize with reference image index not _REF.",
            "reasoning": "Step-by-step reasoning why this task type and strategies were chosen.",
            "score_analysis": "Interpretation of each score and threshold violations.",
            "keyword_analysis": "Which keywords are crucial/missing and how they influence strategy choice.",
            "confidence": 0.0
        }}

        ## Few-Shot Examples

        ### Example 1 (text_image_to_image; hard IP)
        Prompt: "Squid Game S3 teaser poster, Gi-hun in a rain-soaked street, neon green mask reflections"
        Output:
        {{
            "task_type": "text_image_to_image",
            "strategies": ["prompt_optimizer", "image_retrieval"],
            "references_needed": ["squid game poster", "gi-hun"],
            "draft_prompt": "The poster based on image 1, a man from image 2 in a rain-soaked street, neon green mask reflections",
            "reasoning": "IP + character likeness + specific aesthetic → needs two references (Gi-hun, official poster style) to anchor identity and tone.",
            "score_analysis": "clip_score low; face_similarity target absent; style_consistency uncertain → retrieval to ground likeness/style.",
            "keyword_analysis": "‘Squid Game’, ‘Gi-hun’, ‘neon mask’ are niche; require grounding.",
            "confidence": 0.93
        }}

        ### Example 2 (text_to_image; generic but descriptive)
        Prompt: "Pixel art of a golden retriever surfing a giant wave at sunset"
        Output:
        {{
            "task_type": "text_to_image",
            "strategies": ["prompt_optimizer"],
            "references_needed": [],
            "draft_prompt": "Pixel art of a golden retriever surfing a giant wave at sunset",
            "reasoning": "No niche entities; text fully specifies subject, action, style.",
            "score_analysis": "semantic_alignment expected adequate; no prior image constraints.",
            "keyword_analysis": "‘pixel art’, ‘retriever’, ‘surfing’, ‘sunset’ are common.",
            "confidence": 0.90
        }}

        ### Example 3 (image_editing_with_prompt; text-only prompt but edit prior image)
        Context: A valid image was just generated (step t-1) of "street portrait, female runner mid-stride".
        Prompt (text-only): "Give it a 90s VHS sitcom vibe with warm halation and grain; keep the same pose and outfit"
        Output:
        {{
            "task_type": "image_editing_with_prompt",
            "strategies": ["prompt_optimizer"],
            "references_needed": [],
            "draft_prompt": "Give it a 90s VHS sitcom vibe with warm halation and grain; keep the same pose and outfit",
            "reasoning": "Text suggests incremental style change to the most recent image while preserving pose/outfit. No specific external reference required.",
            "score_analysis": "prior_image_available=true; content_alignment_high=0.86; style_mismatch=0.41; edit_intent_detected=true → style-only edit is appropriate.",
            "keyword_analysis": "‘90s VHS’, ‘grain’, ‘halation’ are style modifiers without named IP → no retrieval.",
            "confidence": 0.95
        }}

        ### Example 4 (image_editing_with_prompt_and_reference; text-only prompt but needs IP/background match)
        # The original image will always be image 1. And there will be only one reference image which is image 2.
        # Only retrieve one reference image.
        Context: A valid image was just generated (step t-1) of "ogre-like character standing in a forest clearing".
        Prompt (text-only): "Keep the current pose and lighting but move her to the Shrek swamp and match the movie’s green tint and fog"
        Output:
        {{
            "task_type": "image_editing_with_prompt_and_reference",
            "strategies": ["prompt_optimizer", "image_retrieval"],
            "references_needed": ["shrek"],
            "draft_prompt": "Keep the current pose and lighting but move her to the green ogre in image 1 and match the movie’s green tint and fog",
            "reasoning": "User wants to retain existing composition but match a specific IP location and look. External visual target needed for accurate palette/props/fog.",
            "score_analysis": "prior_image_available=true; content_alignment_high=0.83; location_specificity=‘Shrek swamp’; style_target=‘movie’s green tint’ → requires one reference to lock scene aesthetics.",
            "keyword_analysis": "‘Shrek swamp’, ‘movie’s green tint’, ‘fog’ → IP-scene keywords necessitate reference.",
            "confidence": 0.96
        }}
        """

    def _parse_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response to extract multi-agent optimization decision."""
        try:
            # Extract JSON from response
            start_idx = llm_response.find("{")
            end_idx = llm_response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = llm_response[start_idx:end_idx]
            decision = json.loads(json_str)

            # Validate required fields for new format
            required_fields = [
                "task_type",
                "reasoning",
                "references_needed",
                "draft_prompt",
            ]
            for field in required_fields:
                if field not in decision:
                    raise ValueError(f"Missing '{field}' field in decision")

            # Validate task_type
            valid_task_types = [
                "text_to_image",
                "text_image_to_image",
                "image_editing_with_prompt",
                "image_editing_with_prompt_and_reference",
            ]
            if decision["task_type"] not in valid_task_types:
                raise ValueError(f"Invalid task_type: {decision['task_type']}")

            # Validate references_needed is a list
            if not isinstance(decision["references_needed"], list):
                raise ValueError("'references_needed' field must be a list")

            # Set defaults for optional fields
            decision.setdefault("confidence", 0.8)

            return decision

        except Exception as e:
            logger.error(f"Failed to parse LLM decision: {e}")
            logger.error(f"LLM response: {llm_response}")
            raise RuntimeError(f"Failed to parse orchestrator decision: {e}")

    def _get_visual_analysis(self, prompt: str, image: Image.Image) -> str:
        """Get visual analysis from the engine for the given image."""
        # Analyze image with LiteLLM
        analysis_prompt = (
            f"Analyze this image and compare it with the text: '{prompt}'. "
            "Focus on: "
            "1) What the text describes well vs. what it misses "
            "2) Any hallucinations or distorted details that don't match the prompt. "
            "3) Any elements that are not shown in the text but should be added."
            "4) Visual enhancements for better generation quality "
            "Be specific about enhancement opportunities that don't conflict with the original intent."
        )

        response = self.engine(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing images and detecting AI-generated artifacts. Provide concise, focused analysis.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_base64_url(image)
                            },
                        },
                    ],
                },
            ],
            max_tokens=150,
            temperature=0.1,
        )

        analysis = response.choices[0].message.content

        logger.debug(f"Visual analysis completed: {len(analysis)} chars")
        return analysis


if __name__ == "__main__":
    orchestrator = Orchestrator()
    results = orchestrator(
        original_prompt="Doge to the moon",
        current_prompt="A cat wearing a space suit on the moon",
        image=Image.open(
            "/ssddata2/data/kyle/projects/ImaGenPO/evaluate/results/benchmark_100_us/ImaGenPO/0001/best_result.png"
        ),
        current_scores={"clip_score": 0.88},
        optimization_history=[],
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
