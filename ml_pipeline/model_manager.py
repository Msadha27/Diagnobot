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
            if model_name == "moondream2":
                model = await self._load_moondream2()

            elif model_name == "derm_cnn":
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

            else:
                raise ValueError(f"Unknown model: {model_name}")

            self.models[model_name] = model
            self.load_status[model_name] = "ready"
            return model

        except Exception as e:
            self.load_status[model_name] = f"error: {e}"
            raise

    # ================= VISION =================

    async def _load_moondream2(self):
        model_id = "vikhyatk/moondream2"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True
        ).to(self.device)

        model.eval()

        return {"model": model, "tokenizer": tokenizer}

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
