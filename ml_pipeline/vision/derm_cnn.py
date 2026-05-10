"""
Dermatology Module
Skin lesion detection using Derm CNN (HAM10000) with real-time webcam support
"""

import logging
import json
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import models, transforms
from pathlib import Path
from threading import Thread
import queue

logger = logging.getLogger(__name__)

LOCAL_CONFIDENCE_THRESHOLD = 0.45


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

    LOCAL_DISEASE_CLASSES = {
        "acanthosis nigricans": {"severity": "medium", "code": "AN"},
        "acne": {"severity": "low", "code": "ACNE"},
        "Bullous": {"severity": "high", "code": "BULL"},
        "candidiasis": {"severity": "medium", "code": "CAND"},
        "Eczema Photos": {"severity": "medium", "code": "ECZ"},
        "impetigo": {"severity": "high", "code": "IMP"},
        "Lupus": {"severity": "high", "code": "LUP"},
        "molluscum-contagiosum": {"severity": "medium", "code": "MC"},
        "nevus": {"severity": "low", "code": "NEV"},
        "Urticaria Hives": {"severity": "medium", "code": "URT"},
    }

    def __init__(self, model_manager: "ModelManager"):
        self.model_manager = model_manager
        self.derm_cnn = None
        self.torch_model = None
        self.torch_transform = None
        self.mobile_model = None
        self.onnx_session = None
        self.onnx_input_name = None
        self.mobile_class_names = []
        self.local_index = []
        self.classifier_mode = "unavailable"
        self.webcam_running = False
        self.webcam_thread = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

    async def initialize(self) -> None:
        """Load the Derm CNN model."""
        logger.info("Initializing DermatologyAnalyzer...")
        if self._load_torch_mobilenet():
            self.classifier_mode = "MobileNetV3Small PyTorch"
            logger.info("PyTorch MobileNet classifier loaded")
            return

        if self._load_onnx_mobilenet():
            self.classifier_mode = "MobileNetV3Small ONNX"
            logger.info("ONNX MobileNet classifier loaded")
            return

        if self._load_tensorflow_mobilenet():
            self.classifier_mode = "MobileNetV3Small"
            logger.info("TensorFlow MobileNet classifier loaded")
            return

        try:
            self.derm_cnn = await self.model_manager.get_model("derm_cnn")
            self.classifier_mode = "Derm CNN"
            logger.info("Derm CNN loaded")
        except Exception as e:
            logger.warning(f"Derm CNN model failed, using local dataset classifier: {e}")
            self.derm_cnn = None
            self._build_local_index()
            self.classifier_mode = "Local Dataset KNN"

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

            if self.torch_model is not None:
                return self._analyze_with_torch_mobilenet(image, image_path, return_detailed)

            if self.onnx_session is not None:
                return self._analyze_with_onnx(image, image_path, return_detailed)

            if self.mobile_model is not None:
                return self._analyze_with_mobilenet(image, image_path, return_detailed)

            if self.derm_cnn is None:
                return self._analyze_with_local_index(image, image_path, return_detailed)

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

            if self.torch_model is not None:
                result = self._analyze_with_torch_mobilenet(image, "webcam_capture", True)
                result["frame_captured"] = True
                return result

            if self.onnx_session is not None:
                result = self._analyze_with_onnx(image, "webcam_capture", True)
                result["frame_captured"] = True
                return result

            if self.mobile_model is not None:
                result = self._analyze_with_mobilenet(image, "webcam_capture", True)
                result["frame_captured"] = True
                return result

            if self.derm_cnn is None:
                result = self._analyze_with_local_index(image, "webcam_capture", True)
                result["frame_captured"] = True
                return result

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

    # ==================== LOCAL LIGHTWEIGHT CLASSIFIER ====================

    def _class_names_path(self) -> Optional[Path]:
        exact = Path("models") / "class_names.json"
        if exact.exists():
            return exact
        matches = sorted(Path("models").glob("class_names*.json"))
        return matches[0] if matches else None

    def _load_class_names(self) -> bool:
        classes_path = self._class_names_path()
        if classes_path is None:
            logger.warning("No class_names.json file found in models/")
            return False
        with open(classes_path, "r", encoding="utf-8") as f:
            self.mobile_class_names = json.load(f)
        return bool(self.mobile_class_names)

    def _load_torch_mobilenet(self) -> bool:
        """Load MobileNet exported from PyTorch Colab training."""
        model_path = Path("models") / "skin_mobilenet.pt"
        if not model_path.exists():
            logger.info("No local PyTorch MobileNet model found in models/")
            return False

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            checkpoint_classes = checkpoint.get("class_names")
            if checkpoint_classes:
                self.mobile_class_names = checkpoint_classes
            elif not self._load_class_names():
                raise ValueError("No class names found in checkpoint or models/class_names.json")

            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, len(self.mobile_class_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.torch_model = model
            self.torch_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            logger.info(f"Loaded PyTorch MobileNet classes: {self.mobile_class_names}")
            return True
        except Exception as e:
            logger.warning(f"PyTorch MobileNet unavailable; falling back: {e}")
            self.torch_model = None
            self.torch_transform = None
            return False

    def _analyze_with_torch_mobilenet(
        self,
        image: Image.Image,
        image_path: str,
        return_detailed: bool,
    ) -> Dict[str, Any]:
        tensor = self.torch_transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.torch_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return self._format_neural_classifier_result(
            probs=probs,
            image=image,
            image_path=image_path,
            return_detailed=return_detailed,
            classifier_model="MobileNetV3Small PyTorch",
        )

    def _load_onnx_mobilenet(self) -> bool:
        """Load MobileNet exported as ONNX for lighter local inference."""
        model_path = Path("models") / "skin_mobilenet.onnx"
        if not model_path.exists():
            logger.info("No local ONNX MobileNet model found in models/")
            return False

        try:
            import onnxruntime as ort

            self.onnx_session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            if not self._load_class_names():
                raise ValueError("class_names.json is empty")

            logger.info(f"Loaded ONNX MobileNet classes: {self.mobile_class_names}")
            return True
        except Exception as e:
            logger.warning(f"ONNX MobileNet unavailable; falling back: {e}")
            self.onnx_session = None
            self.onnx_input_name = None
            return False

    def _analyze_with_onnx(
        self,
        image: Image.Image,
        image_path: str,
        return_detailed: bool,
    ) -> Dict[str, Any]:
        resized = image.resize((224, 224))
        batch = np.expand_dims(np.asarray(resized).astype(np.float32), axis=0)
        probs = self.onnx_session.run(None, {self.onnx_input_name: batch})[0][0]
        return self._format_neural_classifier_result(
            probs=probs,
            image=image,
            image_path=image_path,
            return_detailed=return_detailed,
            classifier_model="MobileNetV3Small ONNX",
        )

    def _load_tensorflow_mobilenet(self) -> bool:
        """Load the trained Keras MobileNet model from models/ if present."""
        model_path = Path("models") / "skin_mobilenet.keras"
        if not model_path.exists():
            logger.info("No local MobileNet model found in models/")
            return False

        try:
            import tensorflow as tf

            self.mobile_model = tf.keras.models.load_model(
                str(model_path),
                compile=False,
                safe_mode=False,
            )
            if not self._load_class_names():
                raise ValueError("class_names.json is empty")

            logger.info(f"Loaded MobileNet classes: {self.mobile_class_names}")
            return True
        except Exception as e:
            logger.warning(f"TensorFlow MobileNet unavailable; falling back: {e}")
            self.mobile_model = None
            self.mobile_class_names = []
            return False

    def _analyze_with_mobilenet(
        self,
        image: Image.Image,
        image_path: str,
        return_detailed: bool,
    ) -> Dict[str, Any]:
        """Run the trained TensorFlow MobileNet classifier."""
        resized = image.resize((224, 224))
        batch = np.expand_dims(np.asarray(resized).astype(np.float32), axis=0)
        probs = self.mobile_model.predict(batch, verbose=0)[0]
        return self._format_neural_classifier_result(
            probs=probs,
            image=image,
            image_path=image_path,
            return_detailed=return_detailed,
            classifier_model="MobileNetV3Small",
        )

    def _format_neural_classifier_result(
        self,
        probs: np.ndarray,
        image: Image.Image,
        image_path: str,
        return_detailed: bool,
        classifier_model: str,
    ) -> Dict[str, Any]:
        """Format MobileNet/ONNX probability outputs into API response."""

        ranked_indices = np.argsort(probs)[::-1]
        top_idx = int(ranked_indices[0])
        top_name = self.mobile_class_names[top_idx]
        confidence = float(probs[top_idx])
        disease_info = self.LOCAL_DISEASE_CLASSES.get(
            top_name,
            {"severity": "unknown", "code": "UNK"},
        )
        is_uncertain = confidence < LOCAL_CONFIDENCE_THRESHOLD
        display_label = "Uncertain" if is_uncertain else top_name
        display_severity = "unknown" if is_uncertain else disease_info["severity"]
        display_code = "UNCERTAIN" if is_uncertain else disease_info["code"]

        result: Dict[str, Any] = {
            "status": "success",
            "image_path": str(image_path),
            "image_size": image.size,
            "classification": {
                "label": display_label,
                "disease": display_label,
                "confidence": confidence,
                "severity": display_severity,
                "code": display_code,
                "top_match": top_name,
            },
            "clinical_advice": (
                self._get_uncertain_advice(confidence)
                if is_uncertain
                else self._get_clinical_advice(top_name, confidence)
            ),
            "classifier_model": classifier_model,
            "classifier_note": (
                "MobileNetV3Small trained on the local skin_diseases dataset. "
                "Use as demo decision support only, not diagnosis."
            ),
            "classifier_warning": (
                "Top class confidence is below threshold; classification is uncertain."
                if is_uncertain
                else None
            ),
        }
        result = {key: value for key, value in result.items() if value is not None}

        if return_detailed:
            result["all_predictions"] = [
                {
                    "disease": self.mobile_class_names[int(idx)],
                    "confidence": float(probs[int(idx)]),
                    "severity": self.LOCAL_DISEASE_CLASSES.get(
                        self.mobile_class_names[int(idx)], {"severity": "unknown"}
                    )["severity"],
                }
                for idx in ranked_indices[:5]
            ]

        logger.info(f"{classifier_model} classification: {display_label} / top match {top_name} ({confidence:.2%})")
        return result

    def _build_local_index(self) -> None:
        """Build a tiny color/texture index from data/skin_diseases."""
        data_dir = Path("data") / "skin_diseases"
        if not data_dir.exists():
            logger.warning(f"Local skin dataset not found: {data_dir}")
            return

        self.local_index = []
        for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            for image_file in list(class_dir.glob("*"))[:35]:
                if image_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                try:
                    image = Image.open(image_file).convert("RGB")
                    self.local_index.append(
                        {
                            "disease": class_dir.name,
                            "features": self._extract_features(image),
                        }
                    )
                except Exception:
                    logger.debug(f"Skipping unreadable local skin image: {image_file}")

        logger.info(f"Local dermatology index built with {len(self.local_index)} samples")

    def _analyze_with_local_index(
        self,
        image: Image.Image,
        image_path: str,
        return_detailed: bool,
    ) -> Dict[str, Any]:
        if not self.local_index:
            return {
                "status": "success",
                "image_path": str(image_path),
                "classification": {
                    "label": "Not classified",
                    "disease": "Not classified",
                    "confidence": 0.0,
                    "severity": "unknown",
                    "code": "NA",
                },
                "classifier_model": "unavailable",
                "classifier_note": "Derm CNN failed and local dataset index is empty.",
            }

        query = self._extract_features(image)
        scored = []
        for item in self.local_index:
            distance = float(np.linalg.norm(query - item["features"]))
            scored.append((distance, item["disease"]))
        scored.sort(key=lambda x: x[0])

        class_scores: Dict[str, float] = {}
        for distance, disease in scored[:9]:
            class_scores[disease] = class_scores.get(disease, 0.0) + 1.0 / (distance + 1e-6)

        ranked = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        total = sum(score for _, score in ranked) or 1.0
        disease, score = ranked[0]
        confidence = float(score / total)
        disease_info = self.LOCAL_DISEASE_CLASSES.get(
            disease,
            {"severity": "unknown", "code": "UNK"},
        )
        is_uncertain = confidence < LOCAL_CONFIDENCE_THRESHOLD
        display_label = "Uncertain" if is_uncertain else disease
        display_severity = "unknown" if is_uncertain else disease_info["severity"]
        display_code = "UNCERTAIN" if is_uncertain else disease_info["code"]

        result: Dict[str, Any] = {
            "status": "success",
            "image_path": str(image_path),
            "image_size": image.size,
            "classification": {
                "label": display_label,
                "disease": display_label,
                "confidence": confidence,
                "severity": display_severity,
                "code": display_code,
                "top_match": disease,
            },
            "clinical_advice": (
                self._get_uncertain_advice(confidence)
                if is_uncertain
                else self._get_clinical_advice(disease, confidence)
            ),
            "classifier_model": "Local Dataset KNN",
            "classifier_note": (
                "Lightweight fallback based on local sample similarity. "
                "Use as demo decision support only, not diagnosis."
            ),
            "classifier_warning": (
                "Top match confidence is below threshold; classification is uncertain."
                if is_uncertain
                else None
            ),
        }
        result = {key: value for key, value in result.items() if value is not None}

        if return_detailed:
            result["all_predictions"] = [
                {
                    "disease": name,
                    "confidence": float(class_score / total),
                    "severity": self.LOCAL_DISEASE_CLASSES.get(
                        name, {"severity": "unknown"}
                    )["severity"],
                }
                for name, class_score in ranked[:5]
            ]

        logger.info(f"Local classification: {display_label} / top match {disease} ({confidence:.2%})")
        return result

    def _get_uncertain_advice(self, confidence: float) -> Dict[str, Any]:
        return {
            "urgency": "REVIEW",
            "recommendation": (
                "Classifier confidence is low. Use the top matches only as visual "
                "similarity hints and review with a clinician."
            ),
            "next_steps": [
                "Do not treat this as a diagnosis",
                "Check symptoms, duration, fever, pain, spreading, bleeding, or discharge",
                "Consult a dermatologist or qualified clinician if symptoms persist or worsen",
            ],
            "confidence_level": "Low confidence",
        }

    def _extract_features(self, image: Image.Image) -> np.ndarray:
        """Small RGB histogram plus basic color statistics."""
        resized = image.resize((96, 96))
        arr = np.asarray(resized).astype(np.float32) / 255.0
        hist_parts = [
            np.histogram(arr[:, :, channel], bins=12, range=(0.0, 1.0), density=True)[0]
            for channel in range(3)
        ]
        stats = np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))])
        features = np.concatenate(hist_parts + [stats]).astype(np.float32)
        norm = np.linalg.norm(features)
        return features / norm if norm else features


# ==================== FACTORY ====================

async def create_dermatology_analyzer(model_manager: "ModelManager") -> DermatologyAnalyzer:
    """Create and initialize a DermatologyAnalyzer."""
    analyzer = DermatologyAnalyzer(model_manager)
    await analyzer.initialize()
    return analyzer
