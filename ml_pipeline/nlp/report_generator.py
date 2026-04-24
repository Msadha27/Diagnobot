"""
Report Generation Module
Generates doctor-readable medical reports using:
- BioGPT  : Medical text generation
- BioBart : Patient input → formal report conversion
- ClinicalT5 : Report summarization
"""

import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates clinical reports from various inputs."""

    def __init__(self, model_manager: "ModelManager"):
        self.model_manager = model_manager
        self.biogpt = None
        self.biobart = None
        self.clinical_t5 = None
        self.device = model_manager.device

    async def initialize(self) -> None:
        """Load all report generation models."""
        logger.info("Initializing ReportGenerator...")
        self.biogpt = await self.model_manager.get_model("biogpt")
        self.biobart = await self.model_manager.get_model("biobart")
        self.clinical_t5 = await self.model_manager.get_model("clinical_t5")
        logger.info("✅ Report generation models loaded")

    # ==================== PUBLIC API ====================

    async def generate_report_from_context(
        self,
        clinical_findings: Dict[str, Any],
        patient_info: Optional[Dict[str, str]] = None,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate a medical report from clinical context.

        Args:
            clinical_findings: Dictionary of findings (from X-ray / dermatology)
            patient_info: Optional patient demographics
            max_length: Max token length of the generated report

        Returns:
            Full report, summary, and metadata
        """
        try:
            logger.info("Generating report from clinical findings...")

            prompt = self._build_report_prompt(clinical_findings, patient_info)
            report_text = await self._generate_with_biogpt(prompt, max_length)
            summary = await self._summarize_with_t5(report_text)

            logger.info("✅ Report generated successfully")
            return {
                "status": "success",
                "full_report": report_text,
                "summary": summary,
                "clinical_findings": clinical_findings,
                "patient_info": patient_info,
                "report_type": "clinical_analysis",
                "generation_model": "BioGPT",
            }

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def convert_patient_input_to_report(
        self,
        patient_input: str,
        symptoms: Optional[List[str]] = None,
        medical_history: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert raw patient description to a formal medical report using BioBart.

        Args:
            patient_input: Raw patient description
            symptoms: List of reported symptoms
            medical_history: Patient medical history

        Returns:
            Formatted medical report
        """
        try:
            logger.info("Converting patient input to formal report...")

            prompt = self._build_patient_to_report_prompt(
                patient_input, symptoms, medical_history
            )
            formal_report = await self._generate_with_biobart(prompt)

            logger.info("✅ Patient input converted to formal report")
            return {
                "status": "success",
                "formal_report": formal_report,
                "original_input": patient_input,
                "extracted_symptoms": symptoms,
                "medical_history": medical_history,
                "report_type": "patient_input_conversion",
                "generation_model": "BioBart",
            }

        except Exception as e:
            logger.error(f"Patient input conversion failed: {e}")
            return {"status": "error", "error": str(e)}

    async def summarize_report(
        self,
        report_text: str,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        Summarize a long medical report using ClinicalT5.

        Args:
            report_text: Long medical report text
            max_length: Max summary token length

        Returns:
            Summarized report with compression stats
        """
        try:
            logger.info("Summarizing medical report...")

            summary = await self._summarize_with_t5(report_text, max_length)
            original_words = len(report_text.split())
            summary_words = len(summary.split())
            ratio = summary_words / original_words if original_words else 0

            logger.info(f"✅ Report summarized (compression: {ratio:.2%})")
            return {
                "status": "success",
                "original_report": report_text,
                "summary": summary,
                "original_length": original_words,
                "summary_length": summary_words,
                "compression_ratio": ratio,
                "model": "ClinicalT5",
            }

        except Exception as e:
            logger.error(f"Report summarization failed: {e}")
            return {"status": "error", "error": str(e)}

    # ==================== PROMPT BUILDERS ====================

    def _build_report_prompt(
        self,
        findings: Dict[str, Any],
        patient_info: Optional[Dict[str, str]],
    ) -> str:
        parts = []

        if patient_info:
            if patient_info.get("age"):
                parts.append(f"Patient age: {patient_info['age']}")
            if patient_info.get("gender"):
                parts.append(f"Gender: {patient_info['gender']}")

        if findings.get("clinical_summary"):
            parts.append(f"Clinical findings: {findings['clinical_summary']}")

        if findings.get("findings"):
            descs = [f["description"] for f in findings["findings"] if "description" in f]
            if descs:
                parts.append(f"Detailed findings: {'. '.join(descs)}")

        return "\n".join(parts) + "\n\nBased on these clinical findings, generate a comprehensive medical report:"

    def _build_patient_to_report_prompt(
        self,
        patient_input: str,
        symptoms: Optional[List[str]],
        medical_history: Optional[str],
    ) -> str:
        parts = [
            "Convert the following patient description into a formal clinical report:\n",
            f"Patient Description: {patient_input}\n",
        ]

        if symptoms:
            parts.append(f"Reported Symptoms: {', '.join(symptoms)}\n")

        if medical_history:
            parts.append(f"Medical History: {medical_history}\n")

        parts.append(
            "\nGenerate a structured clinical report with:"
            "\n- Chief Complaint\n- History of Present Illness\n- Symptoms Review"
            "\n- Assessment and Clinical Impression"
        )

        return "".join(parts)

    # ==================== MODEL INFERENCE ====================

    async def _generate_with_biogpt(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using BioGPT."""
        try:
            tokenizer = self.model_manager.tokenizers.get("biogpt")
            if not tokenizer:
                tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
                self.model_manager.tokenizers["biogpt"] = tokenizer

            inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.biogpt.generate(
                    inputs,
                    max_length=max_length,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                )

            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"BioGPT generation failed: {e}")
            return "Error generating report"

    async def _generate_with_biobart(self, prompt: str) -> str:
        """Generate text using BioBart."""
        try:
            tokenizer = self.model_manager.tokenizers.get("biobart")
            if not tokenizer:
                tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")
                self.model_manager.tokenizers["biobart"] = tokenizer

            inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.biobart.generate(
                    inputs,
                    max_length=256,
                    min_length=50,
                    num_beams=4,
                    early_stopping=True,
                )

            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"BioBart generation failed: {e}")
            return "Error converting patient input"

    async def _summarize_with_t5(self, text: str, max_length: int = 256) -> str:
        """Summarize text using ClinicalT5."""
        try:
            tokenizer = self.model_manager.tokenizers.get("clinical_t5")
            if not tokenizer:
                tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-large")
                self.model_manager.tokenizers["clinical_t5"] = tokenizer

            prompt = f"summarize: {text}"
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.clinical_t5.generate(
                    inputs,
                    max_length=max_length,
                    min_length=30,
                    num_beams=4,
                    early_stopping=True,
                )

            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.error(f"ClinicalT5 summarization failed: {e}")
            return text[:256]  # Fallback: truncate


# ==================== FACTORY ====================

async def create_report_generator(model_manager: "ModelManager") -> ReportGenerator:
    """Create and initialize a ReportGenerator."""
    generator = ReportGenerator(model_manager)
    await generator.initialize()
    return generator
