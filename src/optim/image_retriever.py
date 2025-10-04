import warnings
import logging
import json
import os
import requests
import base64
import io
import time
from typing import Dict, List, Optional, Any, Union, Tuple

from PIL import Image
from diffusers.utils import load_image

from .base_llm_engine import BaseLLMEngine

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ImageRetriever(BaseLLMEngine):
    """
    LLM-based Image Retriever that determines optimal image selection and scaling.

    This optimizer analyzes prompts and keywords to decide:
    1. How many reference images to use (1 or multiple)
    2. What search queries to use for finding images
    3. Individual scaling factors for each image
    """

    def __call__(
        self, original_prompt: str, reasoning: str, queries: List[str], **kwargs
    ) -> Dict[str, Any]:
        logger.info(
            "--------------------------------Image Retrieval Agent called --------------------------------"
        )

        # Input validation with assertions
        assert isinstance(original_prompt, str) and original_prompt.strip(), (
            "original_prompt must be a non-empty string"
        )
        assert isinstance(reasoning, str) and reasoning.strip(), (
            "reasoning must be a non-empty string"
        )
        assert isinstance(queries, list) and len(queries) > 0, (
            "queries must be a non-empty list"
        )
        assert all(isinstance(q, str) and q.strip() for q in queries), (
            "all queries must be non-empty strings"
        )

        logger.info("Input validation passed:")
        logger.info(f"  - original_prompt: '{original_prompt}'")
        logger.info(f"  - reasoning: '{reasoning}'")
        logger.info(f"  - queries: {queries}")

        selected_images = []
        successful_queries = []

        for query in queries:
            logger.info(f"Fetching images for query: {query}")
            candidate_images = self._fetch_images(query=query, max_candidates=5)

            # If no images found, use LLM to modify the query
            if not candidate_images:
                logger.info(
                    f"No images found for query '{query}', using LLM to modify query"
                )
                modified_query = self._modify_query_with_llm(
                    query, original_prompt
                )
                logger.info(f"LLM suggested modified query: '{modified_query}'")

                if modified_query and modified_query != query:
                    candidate_images = self._fetch_images(
                        query=modified_query, max_candidates=5
                    )
                    if candidate_images:
                        query = modified_query  # Use the modified query for the rest of the process
                        logger.info(
                            f"Successfully found {len(candidate_images)} images with modified query"
                        )
                    else:
                        logger.error(
                            f"No images found even with modified query '{modified_query}' for original query '{query}'"
                        )
                        raise RuntimeError(
                            f"Failed to find any images for query '{query}' or its modified version '{modified_query}'"
                        )
                else:
                    logger.error(
                        f"LLM could not suggest a better query for '{query}'"
                    )
                    raise RuntimeError(
                        f"Failed to find images for query '{query}' and LLM could not suggest alternative"
                    )

            analysis_prompt = self._create_prompt(
                original_prompt=original_prompt,
                query=query,
                candidate_images=candidate_images,
                category="general",
                max_selections=1,
            )

            # Prompt LLM
            content = [{"type": "text", "text": analysis_prompt}]

            # Add images to the content array
            for i, image in enumerate(candidate_images):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_to_base64_url(image)},
                    }
                )

            response = self.engine(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
            )

            llm_response = response.choices[0].message.content.strip()
            selected_image, reasons = self._parse_response(
                llm_response, candidate_images
            )
            assert selected_image and len(selected_image) > 0, (
                f"No images were selected for query '{query}'"
            )
            selected_images.append(selected_image[0])
            successful_queries.append(query)
            logger.info(
                f"Selected {len(selected_image)} images for query: '{query}'"
            )

        # Final output validation
        assert len(selected_images) > 0, (
            "No images were successfully selected for any query"
        )
        # assert len(successful_queries) == len(selected_images), "Mismatch between selected images and successful queries"
        print(selected_images)

        result = {
            "images": selected_images,
            "queries": successful_queries,
            "reasoning": reasons,
        }

        logger.info("IP-Adapter optimizer output:")
        logger.info(f"  - Total images selected: {len(selected_images)}")
        logger.info(f"  - Successful queries: {successful_queries}")
        logger.info(f"  - Reasoning: {reasoning}")

        return result

    def _create_prompt(
        self,
        original_prompt: str,
        query: str,
        candidate_images: List[Image.Image],
        category: str,
        max_selections: int = 2,
    ) -> str:
        """
        Use LLM with visual analysis to evaluate and select the best images from candidates.

        Args:
            candidate_images: List of candidate images to evaluate
            query: The search query used to find these images
            original_prompt: The original text prompt for generation
            category: The category (content, style, context) this query belongs to
            max_selections: Maximum number of images to select

        Returns:
            List of tuples (selected_image, relevance_score)
        """

        # Create analysis prompt
        return f"""
        You are an expert visual analyst evaluating reference images for text-to-image generation. 

        CONTEXT:
        - Original prompt: "{original_prompt}"
        - Search query: "{query}"
        - Category: {category}
        - Purpose: Select the best reference images to guide AI image generation
        - You must select at least one image.
        
        TASK:
        Analyze each provided image and evaluate how well it matches the search query and would help generate the target prompt.

        For {category} category:
        - CONTENT: Look for objects, people, locations, compositions that match the query
        - STYLE: Look for artistic styles, visual aesthetics, color palettes, techniques
        - CONTEXT: Look for environmental context, mood, atmosphere, setting details

        EVALUATION CRITERIA:
        1. **Query Match**: How well does the image match the specific search query?
        2. **Visual Quality**: Is the image clear, well-composed, and visually appealing?
        3. **Usefulness**: Would this image provide good visual guidance for AI generation?
        4. **Distinctiveness**: Does it offer unique visual information not found in other candidates?

        INSTRUCTIONS:
        - Rate each image from 0.0 to 1.0 (higher = better)
        - Select up to {max_selections} best images
        - Provide brief reasoning for each selection

        Respond with ONLY a JSON object in the following format (this is an example):
        {{
            "selections": [
                {{
                    "image_index": 0,
                    "score": 0.85,
                    "reasoning": "Excellent match for query, high visual quality, provides clear guidance"
                }},
                {{
                    "image_index": 1,
                    "score": 0.72,
                    "reasoning": "Good secondary option with different angle/perspective"
                }}
            ]
        }}

        Only include images you would actually select (score >= 0.6). 
        If you are not sure about the images, you can select multiple images. Low scores are allowed.
        """

    def _parse_response(
        self, llm_response: str, candidate_images: List[Image.Image]
    ) -> Tuple[List[Tuple[Image.Image, float]], List[str]]:
        assert isinstance(llm_response, str) and llm_response.strip(), (
            "LLM response must be a non-empty string"
        )
        assert (
            isinstance(candidate_images, list) and len(candidate_images) > 0
        ), "candidate_images must be a non-empty list"

        # Clean and parse JSON
        if llm_response.startswith("```json"):
            llm_response = llm_response[7:]
        if llm_response.endswith("```"):
            llm_response = llm_response[:-3]

        try:
            analysis_result = json.loads(llm_response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"LLM response: {llm_response}")
            raise RuntimeError(f"Invalid JSON response from LLM: {e}")

        selections = analysis_result["selections"]
        if not selections:
            logger.error("LLM did not select any images")
            logger.info(f"LLM response: {llm_response}")
            raise RuntimeError("LLM analysis returned no image selections")

        # Convert to list of (image, score) tuples
        selected_images = []
        reasons = []
        # Sort selections by score descending before picking images
        sorted_selections = sorted(
            selections, key=lambda s: s.get("score", 0.0), reverse=True
        )
        for selection in sorted_selections:
            img_idx = selection.get("image_index")
            score = selection.get("score", 0.0)
            reasoning = selection.get("reasoning", "")
            reasons.append(reasoning)
            if img_idx is not None and 0 <= img_idx < len(candidate_images):
                selected_images.append((candidate_images[img_idx], score))
                # logger.info(f"Selected image {img_idx} (score: {score:.2f}): {reasoning}")

        if not selected_images:
            logger.error("No valid images were selected from candidates")
            raise RuntimeError(
                "Failed to select any valid images from candidates"
            )

        # logger.info(f"Selected {len(selected_images)} images from {len(candidate_images)} candidates")
        return selected_images, reasons

    def _modify_query_with_llm(
        self, original_query: str, original_prompt: str
    ) -> str:
        """Use LLM to modify a search query when no images are found."""
        assert isinstance(original_query, str) and original_query.strip(), (
            "original_query must be a non-empty string"
        )
        assert isinstance(original_prompt, str) and original_prompt.strip(), (
            "original_prompt must be a non-empty string"
        )

        logger.info(f"Asking LLM to modify query: '{original_query}'")

        modification_prompt = f"""
        You are an expert at creating image search queries. A search query failed to return any images from an image search API.

        CONTEXT:
        - Original text prompt: "{original_prompt}"
        - Failed search query: "{original_query}"
        - Goal: Find reference images to help generate the target prompt

        TASK:
        Create a better, more searchable query that is likely to return relevant images. Consider:

        1. **Simplify complex terms**: Replace uncommon/specific terms with more common alternatives
        2. **Add descriptive keywords**: Include visual descriptors that would help find relevant images
        3. **Use popular terms**: Replace niche concepts with mainstream equivalents
        4. **Consider synonyms**: Use alternative words that mean the same thing
        5. **Focus on visual elements**: Emphasize what the image should look like rather than abstract concepts

        EXAMPLES:
        - "Dr Strange" → "Marvel superhero with cape" or "sorcerer with magic"
        - "backroom" → "yellow fluorescent office space" or "liminal empty rooms"
        - "cyberpunk hacker" → "futuristic computer user neon lights"
        - "medieval knight" → "armored warrior with sword"

        Respond with ONLY the modified search query, nothing else. Make it 2-6 words that would likely return relevant images.
        """

        response = self.engine(
            model=self.model,
            messages=[{"role": "user", "content": modification_prompt}],
            temperature=0.3,
        )

        modified_query = response.choices[0].message.content.strip()

        # Clean up the response (remove quotes, extra text, etc.)
        modified_query = modified_query.strip("\"'").strip()

        # Validate the response is reasonable
        if len(modified_query) > 100 or len(modified_query.split()) > 10:
            logger.error(f"LLM response too long: '{modified_query}'")
            raise RuntimeError(
                f"LLM generated invalid query modification: too long ({len(modified_query)} chars)"
            )

        if not modified_query:
            logger.error("LLM returned empty query modification")
            raise RuntimeError("LLM failed to generate query modification")

        return modified_query

    def _fetch_images(
        self, query: str, max_candidates: int = 5
    ) -> List[Image.Image]:
        """Fetch multiple candidate images using RapidAPI Image Search for a given query."""
        assert isinstance(query, str) and query.strip(), (
            "query must be a non-empty string"
        )
        assert isinstance(max_candidates, int) and max_candidates > 0, (
            "max_candidates must be a positive integer"
        )

        # logger.info(f"Fetching candidate images for query: '{query}' (max: {max_candidates})")

        # Validate API key
        api_key = os.getenv("RAPIDAPI_KEY")
        if not api_key:
            logger.error("RAPIDAPI_KEY environment variable not set")
            raise RuntimeError("RAPIDAPI_KEY environment variable is required")

        params = {
            "query": query,
            "limit": max_candidates,  # Fetch more to account for failed loads
            "size": "any",  # Only large, high-resolution images
            "color": "any",
            "type": "any",
            "time": "any",
            "usage_rights": "any",
            "file_type": "any",  # JPG typically better quality than other formats
            "aspect_ratio": "any",
            "safe_search": "off",
            "region": "us",
        }
        headers = {
            "x-rapidapi-host": "real-time-image-search.p.rapidapi.com",
            "x-rapidapi-key": api_key,
        }

        resp = requests.get(
            "https://real-time-image-search.p.rapidapi.com/search",
            headers=headers,
            params=params,
            timeout=60,  # Increased from 10 to 60 seconds
        )
        resp.raise_for_status()

        data = resp.json()
        results = data.get("data") or []

        if not results:
            logger.warning(f"No images found for query '{query}'")
            return []

        # Collect multiple candidate images
        candidate_images = []
        failed_loads = 0

        for result in results:
            if len(candidate_images) >= max_candidates:
                break

            url = result.get("url")
            if url:
                try:
                    image = load_image(url)
                    candidate_images.append(image)
                except Exception as e:
                    failed_loads += 1
                    logger.warning(f"Failed to load image from {url}: {e}")
                    continue

        if failed_loads > 0:
            logger.info(
                f"Failed to load {failed_loads} images out of {len(results)} results"
            )

        logger.info(
            f"Successfully loaded {len(candidate_images)} candidate images for query '{query}'"
        )
        return candidate_images


if __name__ == "__main__":
    ip_adapter_optimizer = ImageRetrieval()
    ip_adapter_optimizer_result = ip_adapter_optimizer(
        original_prompt="Dr Strange in backroom",
        reasoning="Dr Strange is not a common concept; backroom is not a common concept; reference required.",
        queries=["Dr Strange", "backroom"],
    )
    print("IP-Adapter Optimizer Results:")
    print(
        f"  - Images: {len(ip_adapter_optimizer_result['images'])} image sets"
    )
    print(f"  - Queries: {ip_adapter_optimizer_result['queries']}")
    print(f"  - Reasoning: {ip_adapter_optimizer_result['reasoning']}")
    print(ip_adapter_optimizer_result)
