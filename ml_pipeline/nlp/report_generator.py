"""
Report Generation Module

Generates doctor-readable medical reports using:

- Llama 3.2 via Ollama : Clinical reasoning and decision-support
- BioGPT              : Medical text generation fallback/legacy support
- BioBart              : Patient input -> formal report conversion
- ClinicalT5           : Report summarization

The Llama reasoning backend is served locally through Ollama.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

import torch
from transformers import AutoTokenizer
from ollama import chat

from config.settings import settings

if TYPE_CHECKING:
    from ml_pipeline.model_manager import ModelManager

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates clinical reports from various inputs."""

    def __init__(self, model_manager: "ModelManager"):
        self.model_manager = model_manager

        # Legacy/optional models.
        self.biogpt = None
        self.biobart = None
        self.clinical_t5 = None

        # Llama 3.2 via Ollama.
        # This is used as a readiness flag rather than a loaded
        # llama.cpp model object.
        self.phi_reasoner = None

        self.device = model_manager.device

    async def initialize(self) -> None:
        """
        Initialize the local Ollama Llama 3.2 reasoning backend.

        Ollama manages the actual model process. We therefore do not
        load a GGUF file through llama.cpp anymore.
        """

        if getattr(settings, "MOCK_MODE", False):
            logger.info(
                "ReportGenerator running in Simulation Mode "
                "(No external API or heavy model needed)"
            )
            self.phi_reasoner = None
            return

        logger.info(
            "Initializing ReportGenerator with Ollama Llama 3.2..."
        )

        try:
            # Verify that the local Ollama server and Llama model
            # are available.
            def verify_ollama() -> None:
                response = chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Reply with exactly: DIAGNOBOT LLAMA READY"
                            ),
                        }
                    ],
                )

                content = response.message.content.strip()

                if not content:
                    raise RuntimeError(
                        "Ollama Llama 3.2 returned an empty response"
                    )

            await self._run_blocking(verify_ollama)

            # Readiness flag.
            self.phi_reasoner = True

            logger.info(
                "Ollama Llama 3.2 reasoning model ready"
            )

        except Exception as exc:
            self.phi_reasoner = None

            logger.warning(
                "Llama 3.2 unavailable; using safe template fallback: "
                f"{exc}"
            )

    # ==================== PUBLIC API ====================

    async def generate_report_from_context(
        self,
        clinical_findings: Dict[str, Any],
        patient_info: Optional[Dict[str, str]] = None,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate a medical report from clinical context.
        """

        try:
            logger.info(
                "Generating report from clinical findings..."
            )

            prompt = self._build_report_prompt(
                clinical_findings,
                patient_info,
            )

            report_text = await self.generate_report(
                prompt,
                max_length,
            )

            # Keep the summary lightweight for the current pipeline.
            summary = (
                report_text[:200] + "..."
                if len(report_text) > 200
                else report_text
            )

            logger.info(
                "Report generated successfully"
            )

            return {
                "status": "success",
                "full_report": report_text,
                "summary": summary,
                "clinical_findings": clinical_findings,
                "patient_info": patient_info,
                "report_type": "clinical_analysis",
                "generation_model": "Llama 3.2 (Ollama)",
            }

        except Exception as exc:
            logger.error(
                f"Report generation failed: {exc}",
                exc_info=True,
            )

            return {
                "status": "error",
                "error": str(exc),
            }

    async def convert_patient_input_to_report(
        self,
        patient_input: str,
        symptoms: Optional[List[str]] = None,
        medical_history: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert raw patient description to a formal medical report
        using BioBart.
        """

        try:
            logger.info(
                "Converting patient input to formal report..."
            )

            prompt = self._build_patient_to_report_prompt(
                patient_input,
                symptoms,
                medical_history,
            )

            formal_report = await self._generate_with_biobart(
                prompt
            )

            logger.info(
                "Patient input converted to formal report"
            )

            return {
                "status": "success",
                "formal_report": formal_report,
                "original_input": patient_input,
                "extracted_symptoms": symptoms,
                "medical_history": medical_history,
                "report_type": "patient_input_conversion",
                "generation_model": "BioBart",
            }

        except Exception as exc:
            logger.error(
                f"Patient input conversion failed: {exc}"
            )

            return {
                "status": "error",
                "error": str(exc),
            }

    async def summarize_report(
        self,
        report_text: str,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        Summarize a long medical report using ClinicalT5.
        """

        try:
            logger.info(
                "Summarizing medical report..."
            )

            summary = await self._summarize_with_t5(
                report_text,
                max_length,
            )

            original_words = len(
                report_text.split()
            )

            summary_words = len(
                summary.split()
            )

            ratio = (
                summary_words / original_words
                if original_words
                else 0
            )

            logger.info(
                f"Report summarized "
                f"(compression: {ratio:.2%})"
            )

            return {
                "status": "success",
                "original_report": report_text,
                "summary": summary,
                "original_length": original_words,
                "summary_length": summary_words,
                "compression_ratio": ratio,
                "model": "ClinicalT5",
            }

        except Exception as exc:
            logger.error(
                f"Report summarization failed: {exc}"
            )

            return {
                "status": "error",
                "error": str(exc),
            }

    # ==================== PROMPT BUILDERS ====================

    def _build_report_prompt(
        self,
        findings: Dict[str, Any],
        patient_info: Optional[Dict[str, str]],
    ) -> str:

        parts = []

        if patient_info:

            if patient_info.get("age"):
                parts.append(
                    f"Patient age: {patient_info['age']}"
                )

            if patient_info.get("gender"):
                parts.append(
                    f"Gender: {patient_info['gender']}"
                )

        if findings.get("clinical_summary"):
            parts.append(
                f"Clinical summary: "
                f"{findings['clinical_summary']}"
            )

        if findings.get("abnormal_labs"):

            lab_lines = []

            for lab in findings["abnormal_labs"][:12]:

                low = lab.get("reference_low")
                high = lab.get("reference_high")

                reference = ""

                if low is not None and high is not None:
                    reference = (
                        f" reference {low:g}-{high:g}"
                    )

                elif high is not None:
                    reference = (
                        f" desirable <= {high:g}"
                    )

                elif low is not None:
                    reference = (
                        f" desirable >= {low:g}"
                    )

                value = self._format_lab_value(
                    lab.get("value")
                )

                lab_lines.append(
                    f"- {lab['name']}: {value} "
                    f"{lab.get('unit', '')} "
                    f"({lab['status'].replace('_', ' ')}, "
                    f"{lab.get('severity', 'unknown')} severity;"
                    f"{reference})"
                )

            parts.append(
                "Abnormal or borderline laboratory values:\n"
                + "\n".join(lab_lines)
            )

        if findings.get("abnormal_labs"):

            suggestion_lines = []

            for lab in findings["abnormal_labs"][:8]:

                suggestions = (
                    lab.get("suggestions") or []
                )

                if suggestions:

                    suggestion_lines.append(
                        f"- {lab['name']} target: "
                        f"{lab.get('target', 'within reference range')}. "
                        f"Suggested actions: "
                        f"{' '.join(suggestions[:3])}"
                    )

            if suggestion_lines:
                parts.append(
                    "Threshold-based suggestions:\n"
                    + "\n".join(suggestion_lines)
                )

        if findings.get("normal_labs"):

            normal_names = [
                (
                    f"{lab['name']} "
                    f"{self._format_lab_value(lab.get('value'))} "
                    f"{lab.get('unit', '')}"
                )
                for lab in findings["normal_labs"][:10]
            ]

            parts.append(
                "Selected normal laboratory values: "
                + ", ".join(normal_names)
            )

        if findings.get("risk_flags"):

            parts.append(
                "Risk flags from extraction: "
                + ", ".join(findings["risk_flags"])
            )

        if findings.get("condition_hints"):

            parts.append(
                "Clinical areas to review: "
                + ", ".join(findings["condition_hints"])
            )

        if findings.get("description"):

            parts.append(
                "Visual analysis description: "
                + findings["description"]
            )

        if findings.get("findings"):

            pathologies = [
                finding["name"]
                for finding in findings["findings"]
                if finding.get("name")
            ]

            if pathologies:
                parts.append(
                    "Detected pathologies: "
                    + ", ".join(pathologies)
                )

        return (
            "\n".join(parts)
            + "\n\n"
            "Based on these extracted findings, generate a "
            "cautious medical decision-support report. "
            "Focus on the lab abnormalities and their clinical "
            "interpretation. Do not invent symptoms that are "
            "not present in the extracted findings."
        )

    @staticmethod
    def _format_lab_value(value: Any) -> str:

        if isinstance(value, (int, float)):
            return f"{value:g}"

        if value is None:
            return "not reported"

        return str(value)

    def _build_patient_to_report_prompt(
        self,
        patient_input: str,
        symptoms: Optional[List[str]],
        medical_history: Optional[str],
    ) -> str:

        parts = [
            "Convert the following patient description "
            "into a formal clinical report:\n",
            f"Patient Description: {patient_input}\n",
        ]

        if symptoms:
            parts.append(
                "Reported Symptoms: "
                + ", ".join(symptoms)
                + "\n"
            )

        if medical_history:
            parts.append(
                f"Medical History: "
                f"{medical_history}\n"
            )

        parts.append(
            "\nGenerate a structured clinical report with:"
            "\n- Chief Complaint"
            "\n- History of Present Illness"
            "\n- Symptoms Review"
            "\n- Assessment and Clinical Impression"
        )

        return "".join(parts)

    async def generate_report(
        self,
        context: str,
        max_length: int = 512,
    ) -> str:
        """
        Unified clinical reasoning method using Llama 3.2
        through Ollama.
        """

        return await self._generate_with_gemma(
            context,
            max_length,
        )

    # ==================== OLLAMA REASONING ====================

    async def _generate_with_gemma(
        self,
        context: str,
        max_length: int = 512,
    ) -> str:
        """
        Generate clinical reasoning using Llama 3.2 via Ollama.

        The method name is preserved for compatibility with
        existing project code.
        """

        if self.phi_reasoner is None:
            return self._generate_safe_fallback(
                context
            )

        try:

            logger.info(
                "Running Llama 3.2 Ollama reasoning..."
            )

            reasoning_tokens = min(
                max_length,
                getattr(
                    settings,
                    "REASONING_MAX_TOKENS",
                    96,
                ),
            )

            prompt = (
                "You write concise, cautious medical "
                "decision-support summaries for clinicians. "
                "Do not claim a definitive diagnosis. "
                "Do not invent symptoms, findings, laboratory "
                "values, or history. "
                "Keep the response to 2-3 sentences.\n\n"
                "Clinical findings:\n"
                f"{context[:700]}"
            )

            def infer() -> str:

                response = chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    options={
                        "temperature": 0.1,
                        "num_predict": reasoning_tokens,
                    },
                )

                return (
                    response.message.content.strip()
                )

            description = await self._run_blocking(
                infer
            )

            if not description:
                raise RuntimeError(
                    "Llama 3.2 returned an empty response"
                )

            return description

        except Exception as exc:

            logger.error(
                f"Llama 3.2 Ollama reasoning failed: {exc}",
                exc_info=True,
            )

            return self._generate_safe_fallback(
                context
            )

    def _generate_safe_fallback(
        self,
        context: str,
    ) -> str:
        """
        Deterministic reasoning fallback when the local LLM
        is unavailable.
        """

        lowered = context.lower()

        urgent_terms = [
            "pneumothorax",
            "severe",
            "bleeding",
            "necrosis",
            "unconscious",
            "chest pain",
            "shortness of breath",
            "difficulty breathing",
            "high fever",
        ]

        risk = (
            "urgent"
            if any(
                term in lowered
                for term in urgent_terms
            )
            else "routine"
        )

        next_step = (
            "Escalate for urgent clinician review now."
            if risk == "urgent"
            else (
                "Review with a clinician and correlate "
                "with symptoms, vitals, and history."
            )
        )

        return (
            "Clinical decision-support summary:\n"
            f"- Input findings: {context[:900]}\n"
            f"- Risk flag: {risk}\n"
            f"- Suggested next step: {next_step}\n"
            "- Safety note: This AI output is not a "
            "diagnosis or treatment plan."
        )

    # ==================== LEGACY MODEL INFERENCE ====================

    async def _generate_with_biogpt(
        self,
        prompt: str,
        max_length: int = 512,
    ) -> str:
        """Generate text using BioGPT."""

        try:

            tokenizer = (
                self.model_manager.tokenizers.get(
                    "biogpt"
                )
            )

            if not tokenizer:

                tokenizer = (
                    AutoTokenizer.from_pretrained(
                        "microsoft/biogpt"
                    )
                )

                self.model_manager.tokenizers[
                    "biogpt"
                ] = tokenizer

            inputs = tokenizer.encode(
                prompt,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():

                outputs = self.biogpt.generate(
                    inputs,
                    max_length=max_length,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                )

            return tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

        except Exception as exc:

            logger.error(
                f"BioGPT generation failed: {exc}"
            )

            return "Error generating report"

    async def _generate_with_biobart(
        self,
        prompt: str,
    ) -> str:
        """Generate text using BioBart."""

        try:

            tokenizer = (
                self.model_manager.tokenizers.get(
                    "biobart"
                )
            )

            if not tokenizer:

                tokenizer = (
                    AutoTokenizer.from_pretrained(
                        "GanjinZero/biobart-base"
                    )
                )

                self.model_manager.tokenizers[
                    "biobart"
                ] = tokenizer

            inputs = tokenizer.encode(
                prompt,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():

                outputs = self.biobart.generate(
                    inputs,
                    max_length=256,
                    min_length=50,
                    num_beams=4,
                    early_stopping=True,
                )

            return tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

        except Exception as exc:

            logger.error(
                f"BioBart generation failed: {exc}"
            )

            return "Error converting patient input"

    async def _summarize_with_t5(
        self,
        text: str,
        max_length: int = 256,
    ) -> str:
        """Summarize text using ClinicalT5."""

        try:

            tokenizer = (
                self.model_manager.tokenizers.get(
                    "clinical_t5"
                )
            )

            if not tokenizer:

                tokenizer = (
                    AutoTokenizer.from_pretrained(
                        "luqh/ClinicalT5-large"
                    )
                )

                self.model_manager.tokenizers[
                    "clinical_t5"
                ] = tokenizer

            prompt = f"summarize: {text}"

            inputs = tokenizer.encode(
                prompt,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():

                outputs = self.clinical_t5.generate(
                    inputs,
                    max_length=max_length,
                    min_length=30,
                    num_beams=4,
                    early_stopping=True,
                )

            return tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

        except Exception as exc:

            logger.error(
                f"ClinicalT5 summarization failed: {exc}"
            )

            return text[:256]

    # ==================== UTILITIES ====================

    @staticmethod
    async def _run_blocking(func):
        """
        Run a synchronous Ollama SDK operation without blocking
        the FastAPI event loop.
        """

        return await asyncio.to_thread(func)


# ==================== FACTORY ====================


async def create_report_generator(
    model_manager: "ModelManager",
) -> ReportGenerator:
    """Create and initialize a ReportGenerator."""

    generator = ReportGenerator(
        model_manager
    )

    await generator.initialize()

    return generator