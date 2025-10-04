from typing import Dict, List, Optional, Union, Any
import json

import litellm
from PIL import Image

from ..utils.logging import get_logger
from .base_llm_engine import BaseLLMEngine
try:
    from ..pipeline.pipeline import EnhancedOptimizationResult
except ImportError:
    EnhancedOptimizationResult = None

logger = get_logger(__name__)


class PromptOptimizer(BaseLLMEngine):
    """LiteLLM-based prompt optimizer using the official LiteLLM library."""

    def __call__(
        self,
        task_type: str,
        current_prompt: str,
        original_prompt: str,
        image: Image.Image = None,
        current_scores: Dict[str, float] = None,
        optimization_history: List[Dict[str, Any]] = None,
        # original_keywords: Optional[List[str]] = None,
        # missing_keywords: Optional[List[str]] = None,
        iteration: Optional[int] = None,
        reasoning: str = "",
        **kwargs,
    ) -> Union[Dict[str, Any], str]:
        """Optimize a prompt using LiteLLM methodology.

        Returns Dict[str, Any] for full optimization results or str for simple prompt optimization.
        """
        logger.info(
            "-------------------------------- Prompt Optimizer called --------------------------------"
        )

        # Handle different call patterns - if called from pipeline with iteration info
        # Check if optimization_history contains EnhancedOptimizationResult objects
        needs_conversion = (
            iteration is not None
            and optimization_history
            and len(optimization_history) > 0
            and hasattr(optimization_history[0], "prompt")
            and hasattr(optimization_history[0], "individual_scores")
            and hasattr(optimization_history[0], "hyperparameters")
        )

        if needs_conversion:
            # Convert optimization_history format for pipeline compatibility
            history = []
            for prev_result in optimization_history:
                history.append(
                    {
                        "prompt": prev_result.prompt,
                        "scores": prev_result.individual_scores,
                        "hyperparameters": prev_result.hyperparameters,
                    }
                )
            optimization_history = history
            current_scores = current_scores or {}

        # Initialize defaults
        # Get visual analysis if image is provided
        visual_analysis = self._get_visual_analysis(
            prompt=original_prompt,
            image=image,
        )
        analysis_prompt = self._create_prompt_optimizer_prompt(
            task_type=task_type,
            current_prompt=current_prompt,
            original_prompt=original_prompt,
            current_scores=current_scores,
            optimization_history=optimization_history,
            reasoning=reasoning,
            visual_analysis=visual_analysis,
        )

        # Optimize using LiteLLM
        response = self.engine(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at optimizing prompts for multimodal (text-to-image) generation systems. Your job is to take a user's prompt and improve it for use with advanced text-to-image models, ensuring clarity, specificity, and alignment with the intended subject, style, and constraints. Focus on maximizing image quality and prompt effectiveness. Respond with clear, actionable improvements only.",
                },
                {"role": "user", "content": analysis_prompt},
            ],
            max_tokens=400,  # Increased to prevent JSON truncation
            temperature=0.3,
        )
        llm_response = response.choices[0].message.content.strip()

        # Parse response
        parsed_response = self._parse_response(llm_response)
        optimized_prompt = parsed_response["optimized_prompt"]
        negative_prompts = parsed_response.get("negative_prompts", "")

        # Return format based on call context
        if iteration is not None:
            logger.info(f"Prompt optimized for iteration {iteration + 1}")
        return {
            "optimized_prompt": optimized_prompt,
            "negative_prompts": negative_prompts,
            "original_scores": current_scores,
            "visual_analysis": visual_analysis,
        }

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
        )

        analysis = response.choices[0].message.content

        logger.debug(f"Visual analysis completed: {len(analysis)} chars")
        return analysis

    def _create_prompt_optimizer_prompt(
        self,
        task_type: str,  # "text_to_image" | "text_image_to_image" | "image_editing"
        current_prompt: str,
        original_prompt: str,
        current_scores: Dict[str, float],
        optimization_history: Optional[List[Dict]] = None,
        reasoning: str = "",
        visual_analysis: str = "",
    ) -> str:
        """
        OmniGen2 Prompt Optimizer Agent system prompt (final).
        Emits exactly three Python variables per call:
        1) prompt: <str>
        2) negative_prompts: <comma-separated string>
        """

        def fmt_keywords(kws):
            if not kws:
                return "None"
            kws = list(dict.fromkeys(kws))  # dedupe
            return ", ".join(kws[:16]) + (
                f" (+{len(kws) - 16} more)" if len(kws) > 16 else ""
            )

        score_summary = (
            ", ".join(f"{k}:{v:.2f}" for k, v in current_scores.items())
            or "None"
        )

        hist_lines = []
        if optimization_history:
            for h in optimization_history[-2:]:
                it = h.get("iteration", "N/A")
                sc = ", ".join(
                    f"{k}:{v:.2f}" for k, v in h.get("scores", {}).items()
                )
                hist_lines.append(f"- Iteration {it}: {sc}")
        history_block = "\n".join(hist_lines) if hist_lines else "None"

        return f"""
        ROLE
        You are the Prompt Optimizer Agent. Rewrite the user's request into a clean, actionable instruction string for the selected task type. 
        Always produce a single JSON object with the following variables:
        1) A single string variable named `prompt`
        2) A `negative_prompts` comma-separated string

        TASK TYPE
        {task_type}

        INPUTS
        - ORIGINAL PROMPT: "{original_prompt}"
        - CURRENT OPTIMIZED PROMPT: "{current_prompt}"
        - VISUAL ANALYSIS : {visual_analysis}
        - CURRENT SCORES: {score_summary}
        - RECENT OPTIMIZATION HISTORY:
        {history_block}
        - ORCHESTRATOR REASONING (why this task type): {reasoning}

        OBJECTIVES
        - Preserve essential subject(s), action/intent, and any crucial style/medium cues.
        - If there are any unclear or ambiguous concepts that the image generator might not know try explaining them in the prompt.
        - Clarify composition, lighting, lens/camera, time-of-day only when helpful.
        - Keep wording compact, natural, and non-contradictory.
        - Append concise negatives if artifacts are likely (e.g., "no text artifacts, natural hands").
        - If a concept is niche/ambiguous (celebrity, brand, rare object/place/style) 
        - Always refer to the reference image(s) with image index in the prompt for higher performance.
        OUTPUT RULES (CHOOSE EXACTLY ONE CASE BASED ON task_type)

        A) text_to_image  →  emit ALL:
            {{
                "prompt": "<refined prompt string>",
                "negative_prompts": "term1, term2, term3",
            }}

        Guidelines:
        - One complete directive (Subject → Action/Intent → Composition → Lighting/Camera → Style/Medium).
        - Rich but controlled descriptors; avoid long enumerations or conflicting specs.

        B) text_image_to_image  →  emit ALL:
        {{
            "prompt": "<composite instruction referencing the reference(s) the image agent will retrieve with image index>"
            "negative_prompts": "term1, term2, term3"
        }}

        Guidelines:
        - Assume the Image Retrieval Agent provides reference image(s) for the niche concept(s).
        - Instruction should state the intended composition/edit/compositing with those references (without inventing paths).
        - For example "Add the cat in image 1 to the background in image 2."
        - Always refer to the reference image(s) with image index in the prompt for higher performance.
        - You must refer to the reference image(s) in the prompt for higher performance.

        C) image_editing_with_prompt  →  emit ALL:
            {{
                "prompt": "<instruction to improve the current image to better match the original prompt>",
                "negative_prompts": "term1, term2, term3",
            }}

        D) image_editing_with_prompt_and_reference  →  emit ALL:
            {{
                "prompt": "<instruction to improve the current image using reference(s) to better match the original prompt>",
                "negative_prompts": "term1, term2, term3",
            }}

        Guidelines for Image Editing:
        - You're improving an EXISTING image to better match the SAME prompt
        - Analyze what's wrong with current image (from scores/visual analysis)
        - For prompt-only editing: focus on lighting, color, style, composition improvements
        - For reference editing: identify specific elements that need external reference (faces, objects, backgrounds)
        - Keep the core subject/scene but improve quality/accuracy

        STYLE HEURISTICS
        - Prioritize: Subject → Action/Intent → Composition → Lighting/Camera → Style/Medium.
        - Use concrete, photography/film/art vocabulary over vague adjectives.
        - Avoid contradictions (e.g., “harsh noon sun” + “soft night ambience”).
        - If scores/history imply distortions, add short negatives (hands, faces, watermarks, banding, text).

        VALIDATE BEFORE EMITTING
        - Instruction matches the declared task_type.
        - No invented file paths or unavailable inputs; rely on image retrieval only via `image_retrieval`.
        - If niche concept and no images: `should_call = True` with ONE best query (optionally add a second only if indispensable).

        —— EXAMPLES (SPLIT BY CASE) ——

        # CASE A: text_to_image
        {{
            "prompt": "The sun rises slightly; clear dew on rose petals; a crystal ladybug crawls toward a dew bead; early-morning garden backdrop; macro lens.",
            "negative_prompts": "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
        }}

        # CASE B1: text_image_to_image  (example prompt: "Dr Strange in backroom")
        {{
            prompt = "Compose a scene with the character (Dr Strange) from image 1 standing in a dim, fluorescent ‘backrooms’ corridor from image 2; center-frame, medium shot; flat overhead lighting, subtle fog; emphasize iconic outfit and cape motion."
            negative_prompts = "text artifacts, over-smoothing, waxy skin, warped hands, banding"
        }}

        
        # CASE B2: text_image_to_image
        {{
            "prompt" = "Place the toy from image 1 into the hands of the person in image 2 in a parking-lot setting; align scale and grip; match lighting direction and color temperature."
            "negative_prompts" = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
        }}


        # CASE C: image_editing_with_prompt (improve existing image)
        Original prompt: "Dr Strange in backroom"
        Current image issues: Low lighting quality, poor color balance
        {{
            "prompt": "Improve the lighting and color balance of the current character (Dr Strange) in backroom scene; enhance contrast and fix dim areas; maintain character pose and backroom atmosphere",
            "negative_prompts": "overexposure, harsh shadows, color banding, washed out colors",
        }}

        # CASE D: image_editing_with_prompt_and_reference (improve with reference)
        # The original image will always be image 1. And there will be one reference image which is image 2.
        Original prompt: "Dr Strange in backroom" 
        Current image issues: Character face doesn't look like Dr Strange
        {{
            "prompt": "Fix the character's face in the current backroom scene to match image 2 (character (Dr Strange)); maintain the existing pose and backroom setting in image 1; improve facial accuracy",
            "negative_prompts": "wrong face, generic face, blurry features, face artifacts",
        }}

        EMIT EXACTLY ONE CASE PER CALL:
        - text_to_image → emit JSON with `prompt`, `negative_prompts`
        - text_image_to_image → emit JSON with `prompt`, `negative_prompts`
        - image_editing_with_prompt → emit JSON with `prompt`, `negative_prompts`
        - image_editing_with_prompt_and_reference → emit JSON with `prompt`, `negative_prompts`
        NO EXTRA TEXT OUTSIDE THE JSON OBJECT.
        """
        # model = "SDXL-Base"
        # # TODO: make this dynamic

        # # Helper function to format keyword lists
        # def format_keywords(keywords: Optional[List[str]], max_display: int = 10) -> str:
        #     if not keywords:
        #         return "None"
        #     keywords_str = ", ".join(keywords[:max_display])
        #     if len(keywords) > max_display:
        #         keywords_str += f" (and {len(keywords) - max_display} more)"
        #     return keywords_str

        # # Format keywords
        # preserve_keywords = format_keywords(original_keywords)
        # add_keywords = format_keywords(missing_keywords)

        # # Build optimization history section
        # history_section = ""
        # if optimization_history and len(optimization_history) >= 2:
        #     history_lines = ["RECENT OPTIMIZATIONS:"]
        #     for hist in optimization_history[-2:]:
        #         iteration = hist.get("iteration", "N/A")
        #         scores = hist.get("scores", {})
        #         score_parts = [f"{method.title()} {score:.2f}" for method, score in scores.items()]
        #         history_lines.append(f"  Iteration {iteration}: {', '.join(score_parts)}")
        #     history_section = "\n".join(history_lines) + "\n"

        # # Format current performance
        # score_parts = [f"{method.title()} {score:.2f}" for method, score in current_scores.items()]
        # current_performance = ', '.join(score_parts)
        # return f"""
        # You are a skilled prompt refiner for text-to-image models.
        # Your job is to transform human-written prompts into extremely concise, general prompts
        # of 5 words or fewer while preserving the main subject and intent.

        # Reasoning: {reasoning}
        # PROMPT TO OPTIMIZE: '{prompt}'
        # VISUAL ANALYSIS: {visual_analysis}
        # CURRENT PERFORMANCE: {current_performance}
        # REQUIRED KEYWORDS TO PRESERVE: {preserve_keywords}
        # REQUIRED KEYWORDS TO ADD: {add_keywords}
        # {history_section}

        # OPTIMIZATION GUIDELINES:

        # TARGET: Condense the prompt into 5 words or fewer.
        # Keep only the essential subject, action, or style.
        # Remove specific locations, technical jargon, and detailed attributes.
        # Focus on high-level description and core meaning.
        # Avoid adjectives unless crucial for meaning.
        # Ensure output is flexible and reusable for multiple contexts.
        # OPTIMIZE FOR {model}: Use clear, recognizable words from its vocabulary.
        # Maximize generality: no overly rare or narrow terms.
        # Prioritize Subject → Style → Atmosphere if possible.

        # SPECIFIC TASKS:
        # 1. Boil down to the key subject.
        # 2. Drop non-essential descriptive details.
        # 3. Keep 3–5 impactful words.
        # 4. Maintain natural language flow.

        # REQUIRED EXAMPLE OUTPUT FORMAT:
        # **Optimized Prompt**: "[mysterious desert creature scene]"
        # **Enhancements Made**:
        # - Reduced from a detailed description to a core phrase
        # - Preserved subject while removing location/time-specific constraints
        # - Produced a reusable, general prompt that fits multiple generations
        # """
        # return f"""
        # You are a skilled prompt refiner for text-to-image models.
        # Your job is to transform human-written prompts into clearer, more general prompts
        # that maintain the original meaning but avoid excessive specificity.

        # Reasoning: {reasoning}
        # PROMPT TO OPTIMIZE: '{prompt}'
        # VISUAL ANALYSIS: {visual_analysis}
        # CURRENT PERFORMANCE: {current_performance}
        # REQUIRED KEYWORDS TO PRESERVE: {preserve_keywords}
        # REQUIRED KEYWORDS TO ADD: {add_keywords}
        # {history_section}

        # OPTIMIZATION GUIDELINES:

        # TARGET: Create prompts that are natural, broad, and model-friendly (65–75 tokens).
        # Preserve core subject and scene but avoid over-constraining details.
        # Focus on clarity, flow, and broad descriptive terms.
        # Generalize overly specific objects, styles, and modifiers.
        # Ensure the prompt works across multiple contexts and is not overly tied to one setting.
        # Keep a logical order: Subject → Action/Scene → Style → Key Details → Atmosphere.
        # Remove overly technical jargon unless critical.
        # OPTIMIZE FOR {model}: Use terms {model} understands, but keep descriptions open-ended.
        # BALANCE: Avoid overloading with adjectives—maintain variety without clutter.
        # CLIP ALIGNMENT: Use recognizable, universal descriptors that improve semantic match.
        # If missing context, infer a general description that would improve the image.

        # SPECIFIC TASKS:
        # 1. Capture the main subject and its environment without over-specifying.
        # 2. Remove niche or overly rare references unless crucial to meaning.
        # 3. Use broadly understood visual terms (e.g., "outdoor scene" instead of "Yosemite valley at dusk").
        # 4. Prefer flexible styles (e.g., "painterly style" or "cinematic lighting") over rare art movements.
        # 5. Keep prompt smooth, readable, and approximately 65–75 tokens.
        # 6. Remove excessive weights or detailed lens/camera jargon unless needed.

        # REQUIRED EXAMPLE OUTPUT FORMAT (USE GENERAL TERMS):
        # **Optimized Prompt**: "[A mysterious creature on a windswept desert, textured sand, expressive features, emerging from dunes, scattered natural elements, dramatic lighting, soft shadows, wide shot, cinematic, high detail, sharp focus, slightly stylized]"
        # **Enhancements Made**:
        # - Replaced overly specific descriptors with broader, reusable language
        # - Simplified style and camera references to work across multiple settings
        # - Preserved atmosphere while making it more adaptable
        # """
        # return f"""
        # You are an expert prompt optimizer for text-to-image models.
        # Text-to-image models take a text prompt as input and generate images depicting the prompt as output.
        # You translate prompts written by humans into better prompts for the text-to-image models.
        # Your answers should be concise and effective.
        # You are called from an orchestrator to optimize a prompt.
        # Reasoning: {reasoning}
        # PROMPT TO OPTIMIZE: '{prompt}'
        # VISUAL ANALYSIS: {visual_analysis}
        # CURRENT PERFORMANCE: {current_performance}
        # REQUIRED KEYWORDS TO PRESERVE: {preserve_keywords}
        # REQUIRED KEYWORDS TO ADD: {add_keywords}
        # {history_section}
        # OPTIMIZATION GUIDELINES:

        # TARGET: Create prompts that are 65-75 tokens (CLIP encoder limit is 77)
        # Preserve ALL original semantic meaning and key elements
        # Based on visual analysis, emphasize the key components
        # Enhance visual quality based on image analysis
        # Use precise, impactful words instead of long phrases
        # Prioritize: Subject → Action → Style → Key Details → Atmosphere
        # Base changes on visual evidence, not assumptions
        # OPTIMIZE FOR {model}: Use vocabulary and concepts that {model} was trained on
        # MAXIMIZE CLIP SCORE: Include descriptive adjectives, artistic styles, and visual details that align with CLIP's understanding
        # {model}-SPECIFIC: Leverage terms like 'photorealistic', 'detailed', 'high quality', 'masterpiece', 'best quality'
        # CLIP ALIGNMENT: Use concrete visual descriptors that CLIP can easily recognize and score highly
        # For any entities or concepts that are not present in the image, use the prompt to generate a description of the entity or concept.
        # For any entities or concepts that are were not in the training of {model}, use the prompt to generate a description of the entity or concept.
        # If the prompt is not clear, use the visual analysis to generate a description of the image.

        # SPECIFIC TASKS:
        # 1. Analyze the visual content and compare with the text
        # 2. Identify areas where the prompt could be more accurate
        # 3. Add missing visual elements that would improve generation
        # 4. Refine style descriptions for clarity and impact
        # 5. Optimize word efficiency:
        # - Replace wordy phrases with concise equivalents
        # - Eliminate redundant adjectives and connecting words
        # - Count estimated tokens and aim for 65-75 total
        # 6. Address any detected hallucinations or distortions
        # REQUIRED EXAMPLE OUTPUT FORMAT CHANGE THE VALUES IN THE EXAMPLE TO MATCH THE PROMPT:
        # **Optimized Prompt**: "[(desert monster:1.3), damp sand clinging, mineral-coated teeth, expressive eyes, emerging from wind-rippled dune, rippled dunes, scattered shells, tiny crabs, directional sunlight, long shadows, (low-angle 35mm:1.15), shallow depth of field, photorealism, high detail, sharp focus, subtle film grain\n]"
        # **Enhancements Made**:
        # - [List specific changes you made]
        # - [Explain why each change improves quality]
        # - [Note any visual improvements added]
        # """

    def _parse_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse the optimization response containing markdown-formatted output."""
        import re

        # Initialize with defaults
        result = {
            "optimized_prompt": "",
            "negative_prompts": "",
        }
        try:
            # Extract JSON from response
            start_idx = llm_response.find("{")
            end_idx = llm_response.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = llm_response[start_idx:end_idx]
            decision = json.loads(json_str)

            # Update result with parsed values
            result["optimized_prompt"] = decision.get("prompt", "")
            result["negative_prompts"] = decision.get("negative_prompts", "")

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {llm_response}")
            raise RuntimeError(f"Failed to parse LLM response: {e}")

        return result

    def _validate_keyword_preservation(
        self, optimized_prompt: str, original_keywords: List[str]
    ) -> List[str]:
        """Validate that original keywords are preserved in the optimized prompt."""
        if not original_keywords:
            return []

        optimized_lower = optimized_prompt.lower()
        return [
            keyword
            for keyword in original_keywords
            if keyword.lower() not in optimized_lower
        ]


if __name__ == "__main__":
    prompt_optimizer = PromptOptimizer()
    prompt_optimizer_result = prompt_optimizer(
        task_type="text_image_to_image",
        current_prompt="Dr Strange in backroom",
        original_prompt="Dr Strange in backroom",
        image=Image.open(
            "/ssddata2/data/kyle/projects/ImaGenPO/evaluate/results/benchmark_100_us/ImaGenPO/0001/best_result.png"
        ),
        current_scores={"clip_score": 0.88},
        optimization_history=[],
        iteration=0,
        reasoning="",
    )
    print("PROMPT")
    print(prompt_optimizer_result["optimized_prompt"])
    print("NEGATIVE PROMPTS")
    print(prompt_optimizer_result["negative_prompts"])
