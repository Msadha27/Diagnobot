"""
Lightweight Medical Image Analyzer
--------------------------------
Uses PaliGemma-3B (CPU-optimized Vision-Language Model) to produce
natural-language descriptions of medical images.
"""

import logging
import torch
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


# ===================== MEDICAL PROMPTS =====================
# These prompts are tuned so the model responds like a radiologist / dermatologist.
# Keep them short — longer prompts slow CPU inference significantly.

XRAY_PROMPT = (
    "You are an expert radiologist. Analyze this chest X-ray image carefully. "
    "Describe: (1) overall lung field appearance, (2) any visible opacities, "
    "consolidations, or infiltrates, (3) cardiac silhouette, (4) any abnormalities. "
    "Be concise and clinically precise. End with a one-line impression."
)

DERM_PROMPT = (
    "You are an expert dermatologist. Analyze this skin lesion image. "
    "Describe: (1) lesion morphology (size, shape, border, color), "
    "(2) any ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution hints), "
    "(3) most likely differential diagnosis, (4) recommended next step. "
    "Be concise and clinically precise."
)

GENERAL_MEDICAL_PROMPT = (
    "You are a medical imaging expert. Describe what you observe in this medical image. "
    "Include: visible structures, any abnormalities, and a brief clinical impression."
)


# ===================== ANALYZER CLASS =====================

class QwenVLAnalyzer:
    """
    Wraps PaliGemma-3B for medical image analysis.

    Usage:
        analyzer = QwenVLAnalyzer(model_manager)
        await analyzer.initialize()                    # loads model on first call
        result = await analyzer.analyze_xray(path)
        result = await analyzer.analyze_skin(path)
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.model = None       # Qwen2VLForConditionalGeneration instance
        self.processor = None   # AutoProcessor instance (tokenizer + image processor)

    async def initialize(self) -> None:
        """
        Lazy-load PaliGemma-3B from model_manager.
        """
        if self.model is not None:
            return

        logger.info("VisionAnalyzer: requesting vision_vlm model from model_manager...")
        self.model = await self.model_manager.get_model("vision_vlm")
        self.processor = self.model_manager.tokenizers.get("vision_vlm")

        if self.model is None or self.processor is None:
            raise RuntimeError("Vision VLM model failed to load.")
        logger.info("VisionAnalyzer: ready ✅")

    # ===================== PUBLIC API =====================

    async def analyze_xray(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a radiologist-style description of a chest X-ray.

        Args:
            image_path: Absolute path to the X-ray image (JPG/PNG).
            extra_context: Optional clinical note to append to the prompt
                           (e.g. "Patient is 45 years old, smoker").

        Returns:
            Dict with keys: status, description, model, image_path
        """
        prompt = XRAY_PROMPT
        if extra_context:
            prompt += f"\n\nAdditional clinical context: {extra_context}"

        return await self._run_inference(
            image_path=image_path,
            prompt=prompt,
            analysis_type="xray",
        )

    async def analyze_skin(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a dermatologist-style description of a skin lesion.

        Args:
            image_path: Absolute path to the skin image (JPG/PNG).
            extra_context: Optional clinical note (e.g. "Patient reports itching").

        Returns:
            Dict with keys: status, description, model, image_path
        """
        prompt = DERM_PROMPT
        if extra_context:
            prompt += f"\n\nAdditional context: {extra_context}"

        return await self._run_inference(
            image_path=image_path,
            prompt=prompt,
            analysis_type="dermatology",
        )

    async def analyze_general(self, image_path: str) -> Dict[str, Any]:
        """
        Fallback: generic medical image analysis when type is unknown.
        """
        return await self._run_inference(
            image_path=image_path,
            prompt=GENERAL_MEDICAL_PROMPT,
            analysis_type="general",
        )

    # ===================== CORE INFERENCE =====================

    async def _run_inference(
        self,
        image_path: str,
        prompt: str,
        analysis_type: str,
        max_new_tokens: int = 300,
    ) -> Dict[str, Any]:
        """
        Internal method: runs Qwen2-VL inference on one image.

        HOW IT WORKS:
        1. Load image with PIL
        2. Build a chat-style message: [{"role": "user", "content": [image, text]}]
        3. Apply the processor's chat template → input_ids + pixel_values tensors
        4. Call model.generate() on CPU
        5. Decode the output tokens back to text
        6. Strip the input prompt from the output (Qwen echoes the prompt)
        """
        await self.initialize()  # no-op if already loaded

        try:
            # --- 1. Load image ---
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Moondream2 inference on: {Path(image_path).name}")

            # --- 2. Run Inference ---
            # Moondream has a helper method for answering questions
            with torch.no_grad():
                image_embeds = self.model.encode_image(image)
                description = self.model.answer_question(image_embeds, prompt, self.processor)

            logger.info(f"Moondream2 description generated ({len(description)} chars)")

            return {
                "status": "success",
                "analysis_type": analysis_type,
                "description": description,
                "model": "Moondream2",
                "image_path": str(image_path),
                "disclaimer": (
                    "AI-generated description. Must be reviewed by a licensed clinician."
                ),
            }

        except Exception as e:
            logger.error(f"QwenVL inference failed: {e}", exc_info=True)
            return {
                "status": "error",
                "analysis_type": analysis_type,
                "error": str(e),
                "image_path": str(image_path),
            }


# ===================== FACTORY =====================

async def create_qwen_vl_analyzer(model_manager) -> QwenVLAnalyzer:
    """
    Factory function — consistent with xray_analyzer.py and derm_cnn.py pattern.
    Model is NOT loaded here; it loads lazily on the first inference call.
    """
    return QwenVLAnalyzer(model_manager)
