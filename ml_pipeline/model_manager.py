"""
Model Manager - Handles loading, caching, and lifecycle of all ML models
Implements lazy loading and GPU memory optimization
"""

import logging
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    pipeline
)
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Centralized model manager for all ML models.
    Handles loading, caching, and memory optimization.
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        model_cache_dir: str = "./models_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.use_gpu = use_gpu
        self.device = device
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.load_status: Dict[str, str] = {}
        
        logger.info(f"ModelManager initialized on device: {self.device}")
        logger.info(f"Cache directory: {self.model_cache_dir}")
    
    async def preload_models(self, models: List[str]) -> None:
        """
        Preload critical models on startup
        
        Args:
            models: List of model names to preload
        """
        logger.info(f"Preloading {len(models)} critical models...")
        
        tasks = []
        for model_name in models:
            task = asyncio.create_task(self.load_model(model_name))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for model_name, result in zip(models, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to preload {model_name}: {result}")
            else:
                logger.info(f"✅ Preloaded: {model_name}")
    
    async def load_model(self, model_name: str) -> Any:
        """
        Load a model by name (lazy loading)
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model object
        """
        # Check if already loaded
        if model_name in self.models:
            return self.models[model_name]
        
        logger.info(f"Loading model: {model_name}")
        self.load_status[model_name] = "loading"
        
        try:
            # Vision models
            if model_name == "moondream2":
                model = await self._load_moondream2()
            
            elif model_name == "derm_cnn":
                model = await self._load_derm_cnn()
            
            elif model_name == "xray_vision":
                model = await self._load_xray_vision()
            
            # NLP models
            elif model_name == "clinical_bert":
                model = await self._load_clinical_bert()
            
            elif model_name == "biogpt":
                model = await self._load_biogpt()
            
            elif model_name == "biobart":
                model = await self._load_biobart()
            
            elif model_name == "clinical_t5":
                model = await self._load_clinical_t5()
            
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            self.models[model_name] = model
            self.load_status[model_name] = "ready"
            logger.info(f"✅ Model loaded: {model_name}")
            
            return model
        
        except Exception as e:
            self.load_status[model_name] = f"error: {str(e)}"
            logger.error(f"Failed to load {model_name}: {str(e)}")
            raise
    
    # ==================== VISION MODELS ====================
    
    async def _load_moondream2(self) -> Any:
        """Load Moondream2 for X-ray analysis"""
        from transformers import AutoModelForCausalLM
        
        model_id = "vikhyatk/moondream2"
        revision = "2025-06-21"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            device_map={"": self.device}
        )
        
        if self.use_gpu:
            model = model.to(self.device)
        
        return model
    
    async def _load_derm_cnn(self) -> Any:
        """Load Dermatology CNN (HAM10000)"""
        try:
            from transformers import pipeline
            
            # Using a vision classification pipeline for skin lesions
            derm_pipeline = pipeline(
                "image-classification",
                model="iamhmh/derm-cnn-ham10000",
                device=0 if self.use_gpu else -1
            )
            
            return derm_pipeline
        
        except Exception as e:
            logger.warning(f"HAM10000 model load failed, using fallback: {e}")
            # Fallback to a general image classifier
            fallback = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                device=0 if self.use_gpu else -1
            )
            return fallback
    
    async def _load_xray_vision(self) -> Any:
        """Load TorchXRayVision for X-ray feature extraction"""
        try:
            # Note: Requires: pip install torchxrayvision
            import torchxrayvision as xrv
            
            model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
            
            if self.use_gpu:
                model = model.to(self.device)
            
            return model
        
        except ImportError:
            logger.warning("torchxrayvision not available, skipping TorchXRayVision")
            return None
    
    # ==================== NLP MODELS ====================
    
    async def _load_clinical_bert(self) -> Any:
        """Load Bio_ClinicalBERT for clinical text understanding"""
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        
        if self.use_gpu:
            model = model.to(self.device)
        
        self.tokenizers["clinical_bert"] = tokenizer
        
        return model
    
    async def _load_biogpt(self) -> Any:
        """Load BioGPT for medical report generation"""
        model_name = "microsoft/biogpt"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.use_gpu:
            model = model.to(self.device)
        
        self.tokenizers["biogpt"] = tokenizer
        
        return model
    
    async def _load_biobart(self) -> Any:
        """Load BioBart for patient input → formal report conversion"""
        model_name = "GanjinZero/biobart-base"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.use_gpu:
            model = model.to(self.device)
        
        self.tokenizers["biobart"] = tokenizer
        
        return model
    
    async def _load_clinical_t5(self) -> Any:
        """Load ClinicalT5 for report summarization"""
        model_name = "luqh/ClinicalT5-large"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.use_gpu:
            model = model.to(self.device)
        
        self.tokenizers["clinical_t5"] = tokenizer
        
        return model
    
    # ==================== UTILITY METHODS ====================
    
    async def get_model(self, model_name: str) -> Any:
        """
        Get a model by name, loading if necessary
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded model object
        """
        if model_name not in self.models:
            return await self.load_model(model_name)
        
        return self.models[model_name]
    
    async def get_status(self) -> Dict[str, str]:
        """Get status of all models"""
        return self.load_status
    
    async def cleanup(self) -> None:
        """Clean up models and free GPU memory"""
        logger.info("Cleaning up models...")
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'cpu'):
                    model.cpu()
                if hasattr(model, 'delete'):
                    model.delete()
                logger.info(f"Cleaned up: {model_name}")
            except Exception as e:
                logger.warning(f"Error cleaning {model_name}: {e}")
        
        self.models.clear()
        self.pipelines.clear()
        self.tokenizers.clear()
        
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        logger.info("✅ Cleanup complete")
    
    def get_device(self) -> str:
        """Get current device (cuda/cpu)"""
        return self.device
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        if not self.use_gpu or not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
            "memory_free_gb": (torch.cuda.get_device_properties(0).total_memory - 
                               torch.cuda.memory_allocated(0)) / 1e9
        }
