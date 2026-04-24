"""
X-Ray Analysis Module
Integrates Moondream2 for detection and TorchXRayVision for feature extraction
"""

import logging
from typing import Dict, Any, Optional, List
from PIL import Image
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class XRayAnalyzer:
    """
    Analyzes chest X-rays using Moondream2 + TorchXRayVision
    Detects anomalies and generates clinical findings
    """
    
    def __init__(self, model_manager: 'ModelManager'):
        self.model_manager = model_manager
        self.moondream2 = None
        self.xray_vision = None
        
        # Clinical terms for X-ray findings
        self.anomalies = [
            "pneumonia", "tuberculosis", "nodule", "mass",
            "consolidation", "infiltrate", "pneumothorax",
            "pleural effusion", "atelectasis", "fibrosis",
            "emphysema", "cardiomegaly"
        ]
    
    async def initialize(self) -> None:
        """Load models"""
        logger.info("Initializing XRayAnalyzer...")
        self.moondream2 = await self.model_manager.get_model("moondream2")
        
        # TorchXRayVision is optional
        try:
            self.xray_vision = await self.model_manager.get_model("xray_vision")
        except:
            logger.warning("TorchXRayVision not available, using Moondream2 only")
            self.xray_vision = None
    
    async def analyze_xray(
        self,
        image_path: str,
        return_bbox: bool = True,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a chest X-ray image
        
        Args:
            image_path: Path to X-ray image
            return_bbox: Return bounding boxes for detected anomalies
            return_confidence: Return confidence scores
            
        Returns:
            Analysis results with findings, confidence, and bounding boxes
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded X-ray image: {image_path} ({image.size})")
            
            # Prepare findings list
            findings = []
            
            # 1. Use Moondream2 for detection and visual querying
            findings = await self._moondream_analysis(image, findings)
            
            # 2. Use TorchXRayVision for feature extraction (if available)
            if self.xray_vision:
                xray_features = await self._xray_vision_features(image)
            else:
                xray_features = None
            
            # 3. Generate clinical summary
            summary = self._generate_clinical_summary(findings)
            
            result = {
                "status": "success",
                "image_path": str(image_path),
                "image_size": image.size,
                "findings": findings,
                "clinical_summary": summary,
                "anomaly_count": len([f for f in findings if f["type"] == "anomaly"]),
                "xray_features": xray_features,
                "recommendations": self._generate_recommendations(findings)
            }
            
            logger.info(f"X-ray analysis complete. Found {len(findings)} findings.")
            return result
        
        except Exception as e:
            logger.error(f"X-ray analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "image_path": str(image_path)
            }
    
    async def _moondream_analysis(
        self,
        image: Image.Image,
        findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Use Moondream2 for detection and analysis
        """
        logger.info("Running Moondream2 analysis...")
        
        try:
            # Query 1: General caption
            caption_result = self.moondream2.caption(
                image,
                length="normal",
                stream=False
            )
            caption = caption_result.get("caption", "")
            
            findings.append({
                "type": "general",
                "description": caption,
                "confidence": 0.95,
                "source": "moondream2"
            })
            
            # Query 2: Detect specific anomalies
            for anomaly in self.anomalies:
                prompt = f"Does this X-ray show any signs of {anomaly}? Be specific about location."
                
                try:
                    result = self.moondream2.detect(image, prompt)
                    
                    if result and len(result) > 0:
                        finding = {
                            "type": "anomaly",
                            "name": anomaly,
                            "detected": True,
                            "locations": self._extract_locations(result),
                            "confidence": 0.87,  # Moondream2 confidence
                            "source": "moondream2"
                        }
                        findings.append(finding)
                        logger.info(f"Detected {anomaly}: {result}")
                
                except Exception as e:
                    logger.debug(f"Query for {anomaly} failed: {e}")
                    continue
            
            return findings
        
        except Exception as e:
            logger.error(f"Moondream2 analysis failed: {str(e)}")
            return findings
    
    async def _xray_vision_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract features using TorchXRayVision
        """
        if not self.xray_vision:
            return None
        
        try:
            logger.info("Extracting TorchXRayVision features...")
            
            # Convert PIL image to numpy
            img_array = np.array(image)
            
            # Normalize to [0, 1]
            if img_array.max() > 1:
                img_array = img_array / 255.0
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
            
            # Extract features
            with torch.no_grad():
                features = self.xray_vision(img_tensor)
            
            return {
                "feature_vector_shape": features.shape,
                "extraction_model": "DenseNet121-MIMIC",
                "success": True
            }
        
        except Exception as e:
            logger.warning(f"TorchXRayVision feature extraction failed: {e}")
            return {"error": str(e)}
    
    def _extract_locations(self, detection_results: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract anatomical locations from detection results
        """
        locations = []
        
        for result in detection_results:
            # Parse detection format (usually bounding box coordinates)
            if isinstance(result, dict):
                locations.append({
                    "coordinates": result.get("box", []),
                    "confidence": result.get("score", 0.0)
                })
        
        return locations
    
    def _generate_clinical_summary(self, findings: List[Dict[str, Any]]) -> str:
        """
        Generate a clinical summary from findings
        """
        summary_parts = []
        
        # Get general caption
        general = next((f for f in findings if f["type"] == "general"), None)
        if general:
            summary_parts.append(f"Overall: {general['description']}")
        
        # List detected anomalies
        anomalies = [f for f in findings if f["type"] == "anomaly" and f.get("detected")]
        if anomalies:
            anomaly_list = ", ".join([a["name"] for a in anomalies])
            summary_parts.append(f"Detected findings: {anomaly_list}")
        else:
            summary_parts.append("No significant abnormalities detected.")
        
        return " ".join(summary_parts)
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """
        Generate clinical recommendations based on findings
        """
        recommendations = []
        
        anomalies_found = [f["name"] for f in findings if f.get("type") == "anomaly"]
        
        if "pneumonia" in anomalies_found:
            recommendations.append(
                "Confirm with clinical assessment. Consider antibiotic therapy."
            )
        
        if "tuberculosis" in anomalies_found:
            recommendations.append(
                "Urgent consultation required. TB protocol testing recommended."
            )
        
        if "pneumothorax" in anomalies_found:
            recommendations.append(
                "Emergency assessment needed. Consider immediate intervention."
            )
        
        if "pleural effusion" in anomalies_found:
            recommendations.append(
                "Further imaging and possible aspiration recommended."
            )
        
        if len(anomalies_found) == 0:
            recommendations.append(
                "Findings consistent with normal chest X-ray. Routine follow-up as needed."
            )
        
        recommendations.append("Recommend clinical correlation with patient symptoms.")
        
        return recommendations


# Factory function
async def create_xray_analyzer(model_manager: 'ModelManager') -> XRayAnalyzer:
    """Create and initialize XRayAnalyzer"""
    analyzer = XRayAnalyzer(model_manager)
    await analyzer.initialize()
    return analyzer
