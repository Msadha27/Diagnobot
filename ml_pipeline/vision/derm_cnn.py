"""
Dermatology Module
Skin lesion detection using Derm CNN (HAM10000) with real-time webcam support
"""

import logging
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from threading import Thread
import queue

logger = logging.getLogger(__name__)


class DermatologyAnalyzer:
    """
    Analyzes skin conditions using Derm CNN (HAM10000).
    Supports both image upload and real-time webcam analysis.
    """

    # HAM10000 disease classes
    DISEASE_CLASSES = {
        0: {"name": "Melanoma", "severity": "urgent", "code": "MEL"},
        1: {"name": "Melanocytic nevus", "severity": "low", "code": "NV"},
        2: {"name": "Basal cell carcinoma", "severity": "high", "code": "BCC"},
        3: {"name": "Actinic keratosis", "severity": "medium", "code": "AK"},
        4: {"name": "Benign keratosis", "severity": "low", "code": "BKL"},
        5: {"name": "Dermatofibroma", "severity": "low", "code": "DF"},
        6: {"name": "Vascular lesion", "severity": "medium", "code": "VASC"},
    }

    def __init__(self, model_manager: "ModelManager"):
        self.model_manager = model_manager
        self.derm_cnn = None
        self.webcam_running = False
        self.webcam_thread = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

    async def initialize(self) -> None:
        """Load the Derm CNN model."""
        logger.info("Initializing DermatologyAnalyzer...")
        self.derm_cnn = await self.model_manager.get_model("derm_cnn")
        logger.info("✅ Derm CNN loaded")

    # ==================== IMAGE ANALYSIS ====================

    async def analyze_skin_image(
        self,
        image_path: str,
        return_detailed: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze a skin lesion image.

        Args:
            image_path: Path to skin image
            return_detailed: Return top-5 predictions

        Returns:
            Classification, confidence, severity, and clinical advice
        """
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded skin image: {image_path}")

            results = self.derm_cnn(image)

            top_result = results[0] if results else None
            if not top_result:
                return {"status": "error", "error": "No valid results from model"}

            label_index = top_result.get("label", "")
            confidence = top_result.get("score", 0.0)

            try:
                class_idx = (
                    int(label_index.split("_")[-1])
                    if "_" in str(label_index)
                    else int(label_index)
                )
            except Exception:
                class_idx = 0

            disease_info = self.DISEASE_CLASSES.get(
                class_idx,
                {"name": "Unknown", "severity": "unknown", "code": "UNK"},
            )

            result: Dict[str, Any] = {
                "status": "success",
                "image_path": str(image_path),
                "image_size": image.size,
                "classification": {
                    "disease": disease_info["name"],
                    "confidence": float(confidence),
                    "severity": disease_info["severity"],
                    "code": disease_info["code"],
                },
                "clinical_advice": self._get_clinical_advice(
                    disease_info["name"], float(confidence)
                ),
            }

            if return_detailed and len(results) > 1:
                result["all_predictions"] = [
                    {
                        "disease": self.DISEASE_CLASSES.get(i, {}).get("name", "Unknown"),
                        "confidence": r["score"],
                        "severity": self.DISEASE_CLASSES.get(i, {}).get("severity", "unknown"),
                    }
                    for i, r in enumerate(results[:5])
                ]

            logger.info(f"Classification: {disease_info['name']} ({float(confidence):.2%})")
            return result

        except Exception as e:
            logger.error(f"Skin image analysis failed: {e}")
            return {"status": "error", "error": str(e), "image_path": str(image_path)}

    # ==================== WEBCAM ====================

    def start_webcam(self, device_id: int = 0) -> Dict[str, Any]:
        """Start real-time webcam stream for skin analysis."""
        if self.webcam_running:
            return {"status": "already_running", "message": "Webcam already active"}

        try:
            self.webcam = cv2.VideoCapture(device_id)

            if not self.webcam.isOpened():
                return {"status": "error", "message": f"Failed to open webcam device {device_id}"}

            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.webcam.set(cv2.CAP_PROP_FPS, 30)

            self.webcam_running = True
            self.webcam_thread = Thread(target=self._webcam_loop, daemon=True)
            self.webcam_thread.start()

            logger.info(f"Webcam started (device {device_id})")
            return {"status": "success", "message": "Webcam stream started", "device_id": device_id}

        except Exception as e:
            logger.error(f"Webcam startup failed: {e}")
            return {"status": "error", "message": str(e)}

    def stop_webcam(self) -> Dict[str, Any]:
        """Stop the active webcam stream."""
        self.webcam_running = False

        if hasattr(self, "webcam") and self.webcam:
            self.webcam.release()

        logger.info("Webcam stopped")
        return {"status": "success", "message": "Webcam stopped"}

    def _webcam_loop(self) -> None:
        """Background thread: continuously read webcam frames."""
        while self.webcam_running:
            ret, frame = self.webcam.read()

            if not ret:
                logger.warning("Failed to read webcam frame")
                continue

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except Exception:
                    pass

    async def capture_and_analyze(self) -> Dict[str, Any]:
        """Capture the latest webcam frame and run skin analysis on it."""
        if not self.webcam_running:
            return {"status": "error", "message": "Webcam not running. Call start_webcam() first."}

        try:
            frame = self.frame_queue.get(timeout=2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            results = self.derm_cnn(image)

            if not results:
                return {"status": "error", "message": "Model inference failed"}

            top_result = results[0]
            confidence = top_result.get("score", 0.0)

            try:
                label = str(top_result.get("label", ""))
                class_idx = int(label.split("_")[-1]) if "_" in label else 0
            except Exception:
                class_idx = 0

            disease_info = self.DISEASE_CLASSES.get(
                class_idx,
                {"name": "Unknown", "severity": "unknown", "code": "UNK"},
            )

            return {
                "status": "success",
                "frame_captured": True,
                "classification": {
                    "disease": disease_info["name"],
                    "confidence": float(confidence),
                    "severity": disease_info["severity"],
                    "code": disease_info["code"],
                },
                "clinical_advice": self._get_clinical_advice(
                    disease_info["name"], float(confidence)
                ),
            }

        except queue.Empty:
            return {"status": "error", "message": "No frames available"}
        except Exception as e:
            logger.error(f"Webcam capture/analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    # ==================== CLINICAL ADVICE ====================

    def _get_clinical_advice(self, disease: str, confidence: float) -> Dict[str, Any]:
        """Generate clinical advice based on diagnosis and confidence."""
        if confidence > 0.85:
            confidence_level = "High confidence"
        elif confidence > 0.70:
            confidence_level = "Moderate confidence"
        else:
            confidence_level = "Low confidence"

        advice_map: Dict[str, Dict[str, Any]] = {
            "Melanoma": {
                "urgency": "URGENT",
                "recommendation": "Immediate dermatology referral required. Do not delay.",
                "next_steps": [
                    "Schedule urgent dermatology appointment",
                    "Avoid sun exposure",
                    "Document changes in lesion",
                ],
            },
            "Basal cell carcinoma": {
                "urgency": "HIGH",
                "recommendation": "Dermatology referral needed. Treatment options available.",
                "next_steps": [
                    "Schedule dermatology consultation",
                    "Discuss treatment options",
                    "Consider dermatologic biopsy",
                ],
            },
            "Actinic keratosis": {
                "urgency": "MEDIUM",
                "recommendation": "Dermatology consultation recommended for management.",
                "next_steps": [
                    "Schedule dermatology visit",
                    "Discuss treatment options (topical, cryotherapy, etc.)",
                    "Use sunscreen SPF 30+",
                ],
            },
            "Melanocytic nevus": {
                "urgency": "LOW",
                "recommendation": "Likely benign. Routine monitoring recommended.",
                "next_steps": [
                    "Document baseline characteristics",
                    "Monitor for changes (ABCDE rule)",
                    "Sun protection measures",
                ],
            },
        }

        advice = advice_map.get(
            disease,
            {
                "urgency": "MEDIUM",
                "recommendation": "Dermatology consultation recommended.",
                "next_steps": ["Schedule consultation with dermatologist"],
            },
        )

        advice["confidence_level"] = confidence_level
        return advice


# ==================== FACTORY ====================

async def create_dermatology_analyzer(model_manager: "ModelManager") -> DermatologyAnalyzer:
    """Create and initialize a DermatologyAnalyzer."""
    analyzer = DermatologyAnalyzer(model_manager)
    await analyzer.initialize()
    return analyzer
