"""
X-Ray Analysis Module
Integrates TorchXRayVision for pathology detection and feature extraction
"""

import logging
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)


class XRayAnalyzer:
    """
    Analyzes chest X-rays using TorchXRayVision DenseNet-121
    """

    def __init__(self, model_manager: "ModelManager"):
        self.model_manager = model_manager
        self.xray_vision = None

    async def initialize(self):
        logger.info("Initializing XRayAnalyzer...")
        
        try:
            self.xray_vision = await self.model_manager.get_model("xray_vision")
        except Exception as e:
            self.xray_vision = None
            logger.warning(f"TorchXRayVision not available: {e}")

    # ================= MAIN =================

    async def analyze_xray(
        self,
        image_path: str,
        return_bbox: bool = False,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a chest X-ray image using TorchXRayVision.
        """
        try:
            image = Image.open(image_path).convert("L")  # X-rays are typically grayscale
            findings = []

            if self.xray_vision:
                findings, xray_features = await self._xray_vision_analysis(image)
            else:
                return {"status": "error", "error": "TorchXRayVision model not loaded."}

            summary = self._generate_clinical_summary(findings)

            return {
                "status": "success",
                "findings": findings,
                "clinical_summary": summary,
                "xray_features": xray_features,
                "recommendations": self._generate_recommendations(findings),
            }

        except Exception as e:
            logger.error(f"XRay analysis error: {e}")
            return {"status": "error", "error": str(e)}

    # ================= TORCHXRAYVISION =================

    async def _xray_vision_analysis(self, image: Image.Image):
        """
        Extract features and pathology predictions using TorchXRayVision
        """
        try:
            logger.info("Running TorchXRayVision analysis...")
            
            # TorchXRayVision expects images scaled to [-1024, 1024] or [0, 255] depending on preprocessing.
            # We follow standard preprocessing for 224x224.
            img = image.resize((224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.float32)
            
            # Simple normalization to [-1024, 1024] range which is common for XRV
            img_array = (img_array / 255.0) * 2048 - 1024
            
            # Add channel and batch dimension: [1, 1, 224, 224]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                outputs = self.xray_vision(img_tensor)
                
            # Convert logits to probabilities using Sigmoid
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            findings = []
            
            if hasattr(self.xray_vision, 'pathologies'):
                for i, pathology in enumerate(self.xray_vision.pathologies):
                    # Skip entries where pathology name is None or empty string
                    # (TorchXRayVision has placeholder None slots in its pathology list)
                    if not pathology or not pathology.strip():
                        continue

                    prob = float(probs[i])
                    # Consider finding "detected" if probability >= 0.5
                    if prob >= 0.5:
                        findings.append({
                            "type": "anomaly",
                            "name": pathology.lower().strip(),
                            "detected": True,
                            "confidence": round(prob, 4),
                            "source": "torchxrayvision"
                        })
            
            xray_features = {
                "shape": str(outputs.shape),
                "model": "DenseNet121",
                "success": True
            }

            return findings, xray_features

        except Exception as e:
            logger.error(f"TorchXRayVision analysis failed: {e}")
            return [], {"error": str(e)}

    # ================= SUMMARY =================

    def _generate_clinical_summary(self, findings: List[Dict[str, Any]]) -> str:
        anomalies = [f["name"] for f in findings if f.get("type", "") == "anomaly" and f.get("detected")]
        
        if anomalies:
            return f"Model highlights possible indications of: {', '.join(anomalies)}"
        return "No specific significant findings detected by the model."

    # ================= RECOMMENDATIONS =================

    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        recommendations = []
        anomalies = [f["name"] for f in findings if f.get("type", "") == "anomaly" and f.get("detected")]

        if "pneumonia" in anomalies:
            recommendations.append("Confirm with clinical assessment. Consider antibiotic therapy.")
        if "tuberculosis" in anomalies:
            recommendations.append("Urgent consultation required. TB protocol testing recommended.")
        if "pneumothorax" in anomalies:
            recommendations.append("Emergency assessment needed. Consider immediate intervention.")
        
        if not anomalies:
            recommendations.append("No immediate action indicated by automated analysis.")
            
        recommendations.append("Result is AI-generated and must be correlated clinically by a physician.")
        return recommendations


# ================= FACTORY =================

async def create_xray_analyzer(model_manager):
    analyzer = XRayAnalyzer(model_manager)
    await analyzer.initialize()
    return analyzer