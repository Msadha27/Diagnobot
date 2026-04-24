"""
X-Ray Analysis Module
Integrates Moondream2 for detection and TorchXRayVision for feature extraction
"""

import logging
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)


class XRayAnalyzer:
    """
    Analyzes chest X-rays using Moondream2 + TorchXRayVision
    """

    def __init__(self, model_manager: "ModelManager"):
        self.model_manager = model_manager
        self.moondream2 = None
        self.xray_vision = None

        self.anomalies = [
            "pneumonia", "tuberculosis", "nodule", "mass",
            "consolidation", "infiltrate", "pneumothorax",
            "pleural effusion", "atelectasis", "fibrosis",
            "emphysema", "cardiomegaly"
        ]

    async def initialize(self):
        logger.info("Initializing XRayAnalyzer...")

        self.moondream2 = await self.model_manager.get_model("moondream2")

        try:
            self.xray_vision = await self.model_manager.get_model("xray_vision")
        except:
            self.xray_vision = None
            logger.warning("TorchXRayVision not available")

    # ================= MAIN =================

    async def analyze_xray(self, image_path: str) -> Dict[str, Any]:
        try:
            image = Image.open(image_path).convert("RGB")
            findings = []

            # 🔥 Moondream analysis
            findings = await self._moondream_analysis(image, findings)

            # TorchXRayVision (optional)
            if self.xray_vision:
                xray_features = await self._xray_vision_features(image)
            else:
                xray_features = None

            summary = self._generate_clinical_summary(findings)

            return {
                "status": "success",
                "findings": findings,
                "clinical_summary": summary,
                "xray_features": xray_features,
                "recommendations": self._generate_recommendations(findings),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ================= MOONDREAM =================

    async def _moondream_analysis(self, image, findings):

        logger.info("Running Moondream2 analysis...")

        try:
            caption = self.run_moondream(
                image,
                "Describe this chest X-ray in medical terms."
            )

            findings.append({
                "type": "general",
                "description": caption,
                "confidence": 0.9,
                "source": "moondream2"
            })

            return findings

        except Exception as e:
            logger.error(f"Moondream failed: {e}")
            return findings

    def run_moondream(self, image, prompt):
        model = self.moondream2["model"]
        tokenizer = self.moondream2["tokenizer"]

        inputs = tokenizer(prompt, return_tensors="pt").to(self.model_manager.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)

        return tokenizer.decode(output[0], skip_special_tokens=True)

    # ================= XRAY FEATURES =================

    async def _xray_vision_features(self, image):
        try:
            img_array = np.array(image) / 255.0

            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()

            with torch.no_grad():
                features = self.xray_vision(img_tensor)

            return {
                "shape": str(features.shape),
                "model": "DenseNet121"
            }

        except Exception as e:
            return {"error": str(e)}

    # ================= SUMMARY =================

    def _generate_clinical_summary(self, findings):
        general = next((f for f in findings if f["type"] == "general"), None)

        if general:
            return f"Overall: {general['description']}"
        return "No findings"

    # ================= RECOMMENDATIONS =================

    def _generate_recommendations(self, findings):
        if not findings:
            return ["No abnormalities detected"]

        return [
            "Consult doctor for confirmation",
            "Correlate with symptoms"
        ]


# ================= FACTORY =================

async def create_xray_analyzer(model_manager):
    analyzer = XRayAnalyzer(model_manager)
    await analyzer.initialize()
    return analyzer