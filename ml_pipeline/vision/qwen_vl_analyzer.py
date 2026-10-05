"""
Ollama Moondream vision analysis module.

The filename and factory name are kept for compatibility with existing routes.

This module uses the locally installed Ollama Moondream model for
image understanding while preserving the existing DiagnoBot API,
analysis methods, fallbacks, and safety framing.
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from ollama import chat

from config.settings import settings

logger = logging.getLogger(__name__)


class QwenVLAnalyzer:
    """
    Medical image description helper backed by local Ollama Moondream.

    The class name is preserved for compatibility with existing routes
    and project code.

    This produces decision-support observations only. Diagnosis and
    treatment decisions must stay with a qualified clinician.
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.model = None
        self.use_fallback = False

    async def initialize(self) -> None:
        """Initialize and verify the local Ollama Moondream backend."""

        if getattr(settings, "MOCK_MODE", False):
            logger.info(
                "Vision analyzer running in Simulation Mode "
                "(No external API or heavy model needed)"
            )
            self.use_fallback = True
            return

        try:
            logger.info(
                "Initializing vision analyzer with Ollama Moondream..."
            )

            # Verify that Ollama and the Moondream model are reachable.
            # The actual image inference is performed later in
            # _run_vision_inference().
            def verify_ollama() -> None:
                response = chat(
                    model="moondream",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Reply with exactly: MOONDREAM READY"
                            ),
                        }
                    ],
                )

                content = response.message.content.strip()

                if not content:
                    raise RuntimeError(
                        "Ollama Moondream returned an empty response"
                    )

            await asyncio.to_thread(verify_ollama)

            logger.info("Ollama Moondream vision analyzer ready")

        except Exception as exc:
            logger.error(
                f"Ollama Moondream failed to initialize: {exc}",
                exc_info=True,
            )
            logger.info(
                "Using simple image-property fallback for vision analysis"
            )
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
            "This is a chest X-ray. Describe only visible findings: anatomy, "
            "image quality, possible abnormal regions, uncertainty, and urgent "
            "red flags. Do not give a final diagnosis."
        )

        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(
            image_path,
            prompt,
            "xray",
        )

    async def analyze_skin(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe visible findings in a skin, rash, or wound image."""

        if self.use_fallback:
            return await self._fallback_skin_analysis(image_path)

        prompt = (
            "Look at this image carefully. "
            "Describe only the visible skin findings. "
            "Mention the location, color, redness, bumps, rash, swelling, wound, or other "
            "clearly visible features. "
            "Do not diagnose or recommend treatment. "
            "Do not output coordinates, numbers, JSON, or bounding boxes. "
            "Write 1 to 3 short sentences in plain words."
        )

        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(
            image_path,
            prompt,
            "dermatology",
        )

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
            "This may be a wound image. First state whether a clear wound "
            "is actually visible. If no clear wound, swelling, bleeding, "
            "discharge, or dark tissue is visible, say that clearly and do "
            "not invent one. If a wound is visible, describe only visible "
            "findings: wound size impression, redness, swelling, discharge/pus, "
            "bleeding, dark tissue, edge condition, surrounding skin color, "
            "and urgent infection or necrosis red flags. Do not give a final "
            "diagnosis."
        )

        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(
            image_path,
            prompt,
            "wound",
        )

    async def analyze_eye(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe visible eye color changes as possible clinical symptoms."""

        if self.use_fallback:
            return await self._fallback_eye_analysis(image_path)

        prompt = (
            "This is an eye image. Describe visible color-related findings "
            "only: redness, yellowing of sclera, pallor, discharge, swelling, "
            "asymmetry, and whether the appearance suggests urgent eye or "
            "systemic review. Do not give a final diagnosis."
        )

        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(
            image_path,
            prompt,
            "eye",
        )

    async def analyze_fever(
        self,
        image_path: str,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe visible fever-related signs from a face/general image."""

        if self.use_fallback:
            result = await self._fallback_general_analysis(image_path)
            result["analysis_type"] = "fever"
            result["description"] += (
                " Fever cannot be confirmed from a normal image; "
                "temperature history is required."
            )
            return result

        prompt = (
            "This is a patient face/general image. Describe visible "
            "supportive signs only, such as flushed face, sweating, "
            "fatigue appearance, dehydration cues, rash, and urgent "
            "red flags. Fever cannot be diagnosed from image alone; "
            "mention that temperature measurement is required."
        )

        if extra_context:
            prompt += f" Context: {extra_context}"

        return await self._run_vision_inference(
            image_path,
            prompt,
            "fever",
        )

    async def analyze_general(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        """Describe a general medical image."""

        if self.use_fallback:
            return await self._fallback_general_analysis(image_path)

        prompt = (
            "Describe the visible medical image findings and uncertainty. "
            "Only describe what can actually be observed in the image. "
            "Do not invent findings and do not provide a definitive diagnosis."
        )

        return await self._run_vision_inference(
            image_path,
            prompt,
            "general",
        )

    async def _run_vision_inference(
        self,
        image_path: str,
        prompt: str,
        analysis_type: str,
    ) -> Dict[str, Any]:
        """
        Run vision inference using the local Ollama Moondream model.

        Ollama accepts the local image path directly, so the previous
        GGUF/base64/image_url pipeline is no longer required.
        """

        try:
            if not os.path.exists(image_path):
                return {
                    "status": "error",
                    "analysis_type": analysis_type,
                    "error": f"Image file not found: {image_path}",
                    "model": self._model_label(),
                }

            # Verify that the file is a valid image before sending it
            # to Ollama.
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as image_error:
                return {
                    "status": "error",
                    "analysis_type": analysis_type,
                    "error": f"Invalid image file: {image_error}",
                    "model": self._model_label(),
                    "image_path": str(image_path),
                }

            vision_tokens = getattr(
                settings,
                "VISION_MAX_TOKENS",
                80,
            )

            def infer() -> str:
                response = chat(
                    model="moondream",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_path],
                        }
                    ],
                    options={
                        "temperature": 0.1,
                        "num_predict": vision_tokens,
                    },
                )

                return response.message.content.strip()

            logger.info(
                f"Running {self._model_label()} "
                f"{analysis_type} inference..."
            )

            description = await asyncio.to_thread(infer)

            if not description:
                raise RuntimeError(
                    "Moondream returned an empty description"
                )

            return {
                "status": "success",
                "analysis_type": analysis_type,
                "description": description,
                "model": self._model_label(),
                "image_path": str(image_path),
                "disclaimer": (
                    "AI-generated medical decision support. This is not "
                    "a diagnosis; consult a qualified clinician."
                ),
            }

        except Exception as exc:
            logger.error(
                f"Vision analysis failed: {exc}",
                exc_info=True,
            )

            if analysis_type == "xray":
                return await self._fallback_xray_analysis(image_path)

            if analysis_type == "dermatology":
                return await self._fallback_skin_analysis(image_path)

            if analysis_type == "wound":
                result = await self._fallback_skin_analysis(image_path)
                result["analysis_type"] = "wound"
                return result

            if analysis_type == "eye":
                return await self._fallback_eye_analysis(image_path)

            if analysis_type == "fever":
                result = await self._fallback_general_analysis(image_path)
                result["analysis_type"] = "fever"
                result["description"] += (
                    " Fever cannot be confirmed from a normal image; "
                    "temperature history is required."
                )
                return result

            return await self._fallback_general_analysis(image_path)

    async def _fallback_xray_analysis(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
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
                f"- Quality estimate: "
                f"{self._estimate_image_quality(brightness, contrast)}\n"
                "- Vision model is unavailable, so no pathology "
                "description was generated."
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
            return self._error_response(
                "xray",
                image_path,
                str(exc),
            )

    async def _fallback_skin_analysis(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        """Fallback skin/wound analysis using color statistics."""

        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)

            red_mean = float(
                np.mean(image_array[:, :, 0])
            )
            green_mean = float(
                np.mean(image_array[:, :, 1])
            )
            blue_mean = float(
                np.mean(image_array[:, :, 2])
            )

            if red_mean > 150 and green_mean < 120:
                color_assessment = "reddish or inflamed appearance"

            elif (
                red_mean > 120
                and blue_mean > 120
                and green_mean < 120
            ):
                color_assessment = "purple or bluish appearance"

            else:
                color_assessment = "mixed coloration"

            description = (
                "Skin/wound fallback analysis:\n"
                f"- Average RGB: R={red_mean:.0f}, "
                f"G={green_mean:.0f}, B={blue_mean:.0f}\n"
                f"- Color impression: {color_assessment}\n"
                "- Vision model is unavailable, so this is not "
                "a clinical description.\n"
                "- Recommend clinician review for concerning or "
                "worsening symptoms."
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
            return self._error_response(
                "dermatology",
                image_path,
                str(exc),
            )

    async def _fallback_general_analysis(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        """Generic fallback analysis."""

        try:
            image = Image.open(image_path)

            return {
                "status": "success",
                "analysis_type": "general",
                "description": (
                    f"Image size: {image.size}. "
                    f"Image mode: {image.mode}. "
                    "Vision model is unavailable, so no medical "
                    "description was generated."
                ),
                "model": "Image-Metadata Fallback",
                "image_path": str(image_path),
            }

        except Exception as exc:
            return self._error_response(
                "general",
                image_path,
                str(exc),
            )

    async def _fallback_eye_analysis(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        """Fallback eye-color analysis using image color balance."""

        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)

            red_mean = float(
                np.mean(image_array[:, :, 0])
            )
            green_mean = float(
                np.mean(image_array[:, :, 1])
            )
            blue_mean = float(
                np.mean(image_array[:, :, 2])
            )

            impressions = []

            if (
                red_mean > green_mean + 25
                and red_mean > blue_mean + 25
            ):
                impressions.append(
                    "red-dominant appearance"
                )

            if (
                red_mean > 145
                and green_mean > 130
                and blue_mean < 105
            ):
                impressions.append(
                    "yellow/warm color cast"
                )

            if not impressions:
                impressions.append(
                    "no strong color dominance detected"
                )

            return {
                "status": "success",
                "analysis_type": "eye",
                "description": (
                    "Eye-color fallback analysis:\n"
                    f"- Average RGB: R={red_mean:.0f}, "
                    f"G={green_mean:.0f}, B={blue_mean:.0f}\n"
                    f"- Color impression: "
                    f"{', '.join(impressions)}\n"
                    "- This cannot diagnose jaundice, anemia, "
                    "conjunctivitis, or other disease."
                ),
                "model": "Color-Statistic Fallback",
                "image_path": str(image_path),
                "note": "Vision model unavailable.",
            }

        except Exception as exc:
            return self._error_response(
                "eye",
                image_path,
                str(exc),
            )

    def _estimate_image_quality(
        self,
        brightness: float,
        contrast: float,
    ) -> str:
        """Estimate basic image quality from brightness and contrast."""

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
        """Return the human-readable vision backend label."""

        if getattr(
            settings,
            "VISION_MODEL_BACKEND",
            "ollama",
        ) == "paligemma":
            return "PaliGemma"

        return "Moondream (Ollama)"

    def _error_response(
        self,
        analysis_type: str,
        image_path: str,
        error: str,
    ) -> Dict[str, Any]:
        """Build a consistent error response."""

        return {
            "status": "error",
            "analysis_type": analysis_type,
            "error": error,
            "image_path": str(image_path),
            "model": self._model_label(),
        }


async def create_qwen_vl_analyzer(
    model_manager,
) -> QwenVLAnalyzer:
    """Create and initialize the compatibility analyzer."""

    analyzer = QwenVLAnalyzer(model_manager)
    await analyzer.initialize()

    return analyzer