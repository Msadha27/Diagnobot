import logging
import torch
from typing import Dict, List, Any
from pathlib import Path
import asyncio

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    pipeline,
)

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(
        self,
        use_gpu: bool = True,
        model_cache_dir: str = "./models_cache",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.use_gpu = use_gpu
        self.device = device
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.load_status: Dict[str, str] = {}

        logger.info(f"Using device: {self.device}")

    async def preload_models(self, models: List[str]):
        # Load sequentially to prevent RAM exhaustion and system lockups
        for m in models:
            await self.load_model(m)

    async def load_model(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]

        logger.info(f"Loading {model_name}")
        self.load_status[model_name] = "loading"

        try:
            if model_name == "derm_cnn":
                model = await self._load_derm_cnn()

            elif model_name == "xray_vision":
                model = await self._load_xray_vision()

            elif model_name == "clinical_bert":
                model = await self._load_clinical_bert()

            elif model_name == "biogpt":
                model = await self._load_biogpt()

            elif model_name == "biobart":
                model = await self._load_biobart()

            elif model_name == "clinical_t5":
                model = await self._load_clinical_t5()

            elif model_name == "qwen_vl":
                model = await self._load_qwen_vl()

            else:
                raise ValueError(f"Unknown model: {model_name}")

            self.models[model_name] = model
            self.load_status[model_name] = "ready"
            return model

        except Exception as e:
            self.load_status[model_name] = f"error: {e}"
            raise

    # ================= VISION =================

    async def _load_derm_cnn(self):
        return pipeline(
            "image-classification",
            model="iamhmh/derm-cnn-ham10000",
            device=0 if self.use_gpu else -1,
        )

    async def _load_xray_vision(self):
        try:
            import torchxrayvision as xrv

            model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
            return model.to(self.device)

        except:
            return None

    async def _load_qwen_vl(self):
        """
        Load Qwen2-VL-2B-Instruct — a CPU-friendly Vision-Language Model.
        Always loaded in float32 on CPU; bfloat16 is unreliable on most CPUs.
        The AutoProcessor bundles the tokenizer + image processor together.
        """
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        logger.info(f"Loading Qwen2-VL from HuggingFace: {model_name}")

        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=str(self.model_cache_dir),
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,   # CPU-safe dtype
            device_map="cpu",
            cache_dir=str(self.model_cache_dir),
        )
        model.eval()

        # Store processor under the same key so the analyzer can retrieve it
        self.tokenizers["qwen_vl"] = processor
        logger.info("Qwen2-VL-2B-Instruct loaded successfully on CPU")
        return model

    # ================= NLP =================

    async def _load_clinical_bert(self):
        name = "emilyalsentzer/Bio_ClinicalBERT"

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)

        self.tokenizers["clinical_bert"] = tokenizer
        return model

    async def _load_biogpt(self):
        name = "microsoft/biogpt"

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).to(self.device)

        model.eval()
        self.tokenizers["biogpt"] = tokenizer
        return model

    async def _load_biobart(self):
        name = "GanjinZero/biobart-base"

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.device)

        model.eval()
        self.tokenizers["biobart"] = tokenizer
        return model

    async def _load_clinical_t5(self):
        name = "luqh/ClinicalT5-large"

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.device)

        model.eval()
        self.tokenizers["clinical_t5"] = tokenizer
        return model

    # ================= UTILS =================

    async def get_model(self, model_name: str):
        return await self.load_model(model_name)

    async def get_status(self) -> dict:
        """
        Returns a summary of which models are loaded and their status.
        Called by the root GET / endpoint in main.py.
        """
        return {
            "device": self.device,
            "loaded_models": list(self.models.keys()),
            "load_status": self.load_status,
            "total_loaded": len(self.models),
        }

    async def cleanup(self) -> None:
        """
        Graceful shutdown — clears model references so Python GC can free RAM.
        Called by the lifespan shutdown hook in main.py.
        """
        logger.info("ModelManager: releasing loaded models from memory...")
        self.models.clear()
        self.tokenizers.clear()
        self.load_status.clear()
        logger.info("ModelManager: cleanup complete")
