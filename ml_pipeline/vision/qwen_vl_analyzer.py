"""
Moondream GGUF vision analysis module.

The filename and factory name are kept for compatibility with existing routes.
Default backend is Moondream GGUF for 4 GB RAM systems; PaliGemma can be enabled
later through settings when running on stronger hardware.
"""

import asyncio
import base64
import io
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from config.settings import settings

logger = logging.getLogger(__name__)


class QwenVLAnalyzer:
    """
    Medical image description helper backed by Moondream GGUF.

    This produces decision-support observations only. Diagnosis and treatment
    decisions must stay with a qualified clinician.
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.model = None
        self.use_fallback = False

    async def initialize(self) -> None:
        """Load the configured vision model via ModelManager."""
        logger.info(f"Initializing vision analyzer with {settings.VISION_MODEL_BACKEND}...")

        try:
            self.model = await self.model_manager.get_model("vision_vlm")
            if self.model is None:
                raise RuntimeError("Vision model returned None")
            logger.info("Vision analyzer ready")
        except Exception as exc:
            logger.error(f"Vision model failed to initialize: {exc}")
            logger.info("Using simple image-property fallback for vision analysis")
            self.use_fallback = True

    async def analyze_xray(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe visible findings in a chest X-ray image."""
        if self.use_fallback:
            return await self._fallback_xray_analysis(image_path)

        prompt = (
            "This is a chest X-ray. Describe only visible findings: anatomy, image "
            "quality, possible abnormal regions, uncertainty, and urgent red flags. "
            "Do not give a final diagnosis."
        )
        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(image_path, prompt, "xray")

    async def analyze_skin(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe visible findings in a skin, rash, or wound image."""
        if self.use_fallback:
            return await self._fallback_skin_analysis(image_path)

        prompt = (
            "Skin image. If no clear skin finding is visible, say so. Otherwise briefly "
            "describe visible color, border, shape, swelling, discharge, bleeding, ABCDE "
            "warning signs if visible, uncertainty, and whether urgent doctor review is "
            "needed. Do not diagnose."
        )
        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(image_path, prompt, "dermatology")

    async def analyze_wound(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe wound appearance and visible infection warning signs."""
        if self.use_fallback:
            result = await self._fallback_skin_analysis(image_path)
            result["analysis_type"] = "wound"
            return result

        prompt = (
            "This may be a wound image. First state whether a clear wound is actually "
            "visible. If no clear wound, swelling, bleeding, discharge, or dark tissue "
            "is visible, say that clearly and do not invent one. If a wound is visible, "
            "describe only visible findings: wound size impression, redness, swelling, "
            "discharge/pus, bleeding, dark tissue, edge condition, surrounding skin "
            "color, and urgent infection or necrosis red flags. Do not give a final "
            "diagnosis."
        )
        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(image_path, prompt, "wound")

    async def analyze_eye(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe visible eye color changes as possible clinical symptoms."""
        if self.use_fallback:
            return await self._fallback_eye_analysis(image_path)

        prompt = (
            "This is an eye image. Describe visible color-related findings only: redness, "
            "yellowing of sclera, pallor, discharge, swelling, asymmetry, and whether the "
            "appearance suggests urgent eye or systemic review. Do not give a final diagnosis."
        )
        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(image_path, prompt, "eye")

    async def analyze_fever(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe visible fever-related signs from a face/general image."""
        if self.use_fallback:
            result = await self._fallback_general_analysis(image_path)
            result["analysis_type"] = "fever"
            result["description"] += " Fever cannot be confirmed from a normal image; temperature history is required."
            return result

        prompt = (
            "This is a patient face/general image. Describe visible supportive signs only, "
            "such as flushed face, sweating, fatigue appearance, dehydration cues, rash, "
            "and urgent red flags. Fever cannot be diagnosed from image alone; mention that "
            "temperature measurement is required."
        )
        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(image_path, prompt, "fever")

    async def analyze_general(self, image_path: str) -> Dict[str, Any]:
        """Describe a general medical image."""
        if self.use_fallback:
            return await self._fallback_general_analysis(image_path)

        prompt = "Describe the visible medical image findings and uncertainty."
        return await self._run_vision_inference(image_path, prompt, "general")

    async def _run_vision_inference(
        self,
        image_path: str,
        prompt: str,
        analysis_type: str,
    ) -> Dict[str, Any]:
        """Run Moondream GGUF inference in a worker thread."""
        try:
            if not os.path.exists(image_path):
                return {
                    "status": "error",
                    "analysis_type": analysis_type,
                    "error": f"Image file not found: {image_path}",
                    "model": self._model_label(),
                }

            image = Image.open(image_path).convert("RGB")
            max_side = 768 if analysis_type == "xray" else 448
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image_url = self._image_to_data_url(image)

            def infer() -> str:
                response = self.model.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                },
                            ],
                        }
                    ],
                    max_tokens=96,
                    temperature=0.1,
                )
                return response["choices"][0]["message"]["content"].strip()

            logger.info(f"Running {self._model_label()} {analysis_type} inference...")
            description = await asyncio.to_thread(infer)
            if not description.strip():
                raise RuntimeError("Vision model returned an empty description")

            return {
                "status": "success",
                "analysis_type": analysis_type,
                "description": description,
                "model": self._model_label(),
                "image_path": str(image_path),
                "disclaimer": (
                    "AI-generated medical decision support. This is not a diagnosis; "
                    "consult a qualified clinician."
                ),
            }

        except Exception as exc:
            logger.error(f"Vision analysis failed: {exc}", exc_info=True)
            if analysis_type == "xray":
                return await self._fallback_xray_analysis(image_path)
            if analysis_type == "dermatology":
                return await self._fallback_skin_analysis(image_path)
            return await self._fallback_general_analysis(image_path)

    async def _fallback_xray_analysis(self, image_path: str) -> Dict[str, Any]:
        """Simple fallback X-ray analysis using image properties."""
        try:
            image = Image.open(image_path).convert("L")
            image_array = np.array(image)
            brightness = float(np.mean(image_array))
            contrast = float(np.std(image_array))

            description = (
                "X-ray fallback analysis:\n"
                f"- Brightness: {brightness:.1f}/255\n"
                f"- Contrast: {contrast:.1f}\n"
                f"- Quality estimate: {self._estimate_image_quality(brightness, contrast)}\n"
                "- Vision model is unavailable, so no pathology description was generated."
            )

            return {
                "status": "success",
                "analysis_type": "xray",
                "description": description,
                "model": "Image-Property Fallback",
                "image_path": str(image_path),
                "note": "Vision model unavailable.",
            }
        except Exception as exc:
            return self._error_response("xray", image_path, str(exc))

    async def _fallback_skin_analysis(self, image_path: str) -> Dict[str, Any]:
        """Fallback skin/wound analysis using color statistics."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            red_mean = float(np.mean(image_array[:, :, 0]))
            green_mean = float(np.mean(image_array[:, :, 1]))
            blue_mean = float(np.mean(image_array[:, :, 2]))

            if red_mean > 150 and green_mean < 120:
                color_assessment = "reddish or inflamed appearance"
            elif red_mean > 120 and blue_mean > 120 and green_mean < 120:
                color_assessment = "purple or bluish appearance"
            else:
                color_assessment = "mixed coloration"

            description = (
                "Skin/wound fallback analysis:\n"
                f"- Average RGB: R={red_mean:.0f}, G={green_mean:.0f}, B={blue_mean:.0f}\n"
                f"- Color impression: {color_assessment}\n"
                "- Vision model is unavailable, so this is not a clinical description.\n"
                "- Recommend clinician review for concerning or worsening symptoms."
            )

            return {
                "status": "success",
                "analysis_type": "dermatology",
                "description": description,
                "model": "Color-Statistic Fallback",
                "image_path": str(image_path),
                "note": "Vision model unavailable.",
            }
        except Exception as exc:
            return self._error_response("dermatology", image_path, str(exc))

    async def _fallback_general_analysis(self, image_path: str) -> Dict[str, Any]:
        """Generic fallback analysis."""
        try:
            image = Image.open(image_path)
            return {
                "status": "success",
                "analysis_type": "general",
                "description": (
                    f"Image size: {image.size}. Image mode: {image.mode}. "
                    "Vision model is unavailable, so no medical description was generated."
                ),
                "model": "Image-Metadata Fallback",
                "image_path": str(image_path),
            }
        except Exception as exc:
            return self._error_response("general", image_path, str(exc))

    async def _fallback_eye_analysis(self, image_path: str) -> Dict[str, Any]:
        """Fallback eye-color analysis using image color balance."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            red_mean = float(np.mean(image_array[:, :, 0]))
            green_mean = float(np.mean(image_array[:, :, 1]))
            blue_mean = float(np.mean(image_array[:, :, 2]))

            impressions = []
            if red_mean > green_mean + 25 and red_mean > blue_mean + 25:
                impressions.append("red-dominant appearance")
            if red_mean > 145 and green_mean > 130 and blue_mean < 105:
                impressions.append("yellow/warm color cast")
            if not impressions:
                impressions.append("no strong color dominance detected")

            return {
                "status": "success",
                "analysis_type": "eye",
                "description": (
                    "Eye-color fallback analysis:\n"
                    f"- Average RGB: R={red_mean:.0f}, G={green_mean:.0f}, B={blue_mean:.0f}\n"
                    f"- Color impression: {', '.join(impressions)}\n"
                    "- This cannot diagnose jaundice, anemia, conjunctivitis, or other disease."
                ),
                "model": "Color-Statistic Fallback",
                "image_path": str(image_path),
                "note": "Vision model unavailable.",
            }
        except Exception as exc:
            return self._error_response("eye", image_path, str(exc))

    def _estimate_image_quality(self, brightness: float, contrast: float) -> str:
        quality = []
        if 80 <= brightness <= 180:
            quality.append("reasonable exposure")
        elif brightness < 80:
            quality.append("possibly underexposed")
        else:
            quality.append("possibly overexposed")

        if contrast > 40:
            quality.append("high contrast")
        elif contrast < 15:
            quality.append("low contrast")

        return ", ".join(quality)

    def _model_label(self) -> str:
        if settings.VISION_MODEL_BACKEND == "paligemma":
            return "PaliGemma"
        return "Moondream2-GGUF"

    def _image_to_data_url(self, image: Image.Image) -> str:
        """Encode a compact JPEG data URL to keep CPU VLM inference responsive."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _error_response(self, analysis_type: str, image_path: str, error: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "analysis_type": analysis_type,
            "error": error,
            "image_path": str(image_path),
            "model": self._model_label(),
        }


async def create_qwen_vl_analyzer(model_manager) -> QwenVLAnalyzer:
    """Create and initialize the compatibility analyzer."""
    analyzer = QwenVLAnalyzer(model_manager)
    await analyzer.initialize()
    return analyzer
