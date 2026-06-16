import logging
import torch
from typing import Dict, List, Any
from pathlib import Path
import asyncio
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    pipeline,
)

from config.settings import settings

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

    async def get_model(self, model_name: str, unload_others: bool = True):
        """
        Get model instance, optionally unloading others to save RAM.
        Critical for 4GB systems.
        """
        if unload_others:
            others = [m for m in self.models.keys() if m != model_name]
            for other in others:
                await self.unload_model(other)

        if model_name in self.models:
            return self.models[model_name]
        return await self.load_model(model_name)

    async def unload_model(self, model_name: str):
        """
        Unload a model from memory and clear GPU/RAM cache.
        """
        if model_name in self.models:
            logger.info(f"Unloading model to free RAM: {model_name}")
            del self.models[model_name]
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Memory cleared after unloading {model_name}")

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

            elif model_name == "vision_vlm":
                if settings.VISION_MODEL_BACKEND == "paligemma":
                    model = await self._load_paligemma_vision()
                else:
                    model = await self._load_moondream_vision()

            elif model_name == "reasoning_phi":
                model = await self._load_reasoning_phi()

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

    async def _load_paligemma_vision(self):
        """
        Load PaliGemma as the primary medical image description model.

        PaliGemma is a general vision-language model, not a certified medical
        diagnostic device. Routes wrap its output as decision support only.
        """
        model_name = settings.PALIGEMMA_MODEL
        logger.info(f"Loading PaliGemma vision model: {model_name}")

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=str(self.model_cache_dir),
            token=settings.HUGGINGFACE_HUB_TOKEN or None,
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=str(self.model_cache_dir),
            torch_dtype=dtype,
            token=settings.HUGGINGFACE_HUB_TOKEN or None,
        ).to(self.device)
        model.eval()

        self.tokenizers["vision_vlm"] = processor
        logger.info("PaliGemma vision model loaded successfully")
        return model

    async def _load_moondream_vision(self):
        """
        Load Moondream2 (GGUF) — Ultra-lightweight vision for 4GB RAM.
        Uses llama-cpp-python for minimal footprint (~1.8GB total download).
        """
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MoondreamChatHandler
        
        repo_id = settings.MOONDREAM_VISION_MODEL
        text_file = settings.MOONDREAM_TEXT_FILE
        proj_file = settings.MOONDREAM_PROJ_FILE
        
        logger.info(f"Downloading Moondream GGUF components from {repo_id}...")
        
        # Download Text Model
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=text_file,
            cache_dir=str(self.model_cache_dir)
        )
        
        # Download Vision (CLIP/MMProj) Model
        clip_path = hf_hub_download(
            repo_id=repo_id,
            filename=proj_file,
            cache_dir=str(self.model_cache_dir)
        )

        chat_handler = MoondreamChatHandler(clip_model_path=clip_path)

        model = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=1024,
            n_threads=4,
            chat_format="moondream",
            verbose=False
        )

        self.tokenizers["vision_vlm"] = None
        logger.info("Moondream GGUF loaded successfully (4GB RAM Mode)")
        return model

    async def _load_vision_vlm(self):
        return await self._load_moondream_vision()

    async def _load_reasoning_phi(self):
        """
        Load Gemma-2-2B (GGUF) for 4GB RAM.
        Replaces Phi-3.5 with Google's superior logic in a tiny footprint.
        """
        from llama_cpp import Llama

        repo_id = settings.GEMMA_REASONING_MODEL
        filename = settings.GEMMA_REASONING_FILE
        
        logger.info(f"Downloading/Loading GGUF Reasoning: {repo_id}/{filename}")
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(self.model_cache_dir)
        )

        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )

        self.tokenizers["reasoning_phi"] = None # Handled internally
        logger.info("Gemma-2 GGUF loaded (4GB RAM Mode)")
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
