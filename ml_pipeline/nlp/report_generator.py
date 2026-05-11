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
        self.phi_reasoner = None
        self.device = model_manager.device

    async def initialize(self) -> None:
        """
        Load only the primary Reasoning Brain (Gemma-2-2B).
        On 4GB RAM systems, we consolidate all tasks into one powerful model 
        to avoid loading multiple heavy BERT/T5/GPT models.
        """
        logger.info("Initializing ReportGenerator (4GB Lean Mode)...")
        try:
            self.phi_reasoner = await self.model_manager.get_model("reasoning_phi")
            logger.info("Google Gemma-2 reasoning model loaded")
        except Exception as e:
            self.phi_reasoner = None
            logger.warning(f"Reasoning model unavailable; using safe template fallback: {e}")

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
            logger.info("Generating report from clinical findings...")

            prompt = self._build_report_prompt(clinical_findings, patient_info)
            
            # Use Gemma-2-2B for superior reasoning
            if self.phi_reasoner:
                report_text = await self._generate_with_gemma(prompt, max_length)
                gen_model = "Gemma-2-2B-IT"
            else:
                report_text = await self._generate_with_biogpt(prompt, max_length)
                gen_model = "BioGPT"
                
            summary = await self._summarize_with_t5(report_text)

            logger.info("✅ Report generated successfully")
            return {
                "status": "success",
                "full_report": report_text,
                "summary": summary,
                "clinical_findings": clinical_findings,
                "patient_info": patient_info,
                "report_type": "clinical_analysis",
                "generation_model": gen_model,
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
            parts.append(f"Clinical summary: {findings['clinical_summary']}")

        if findings.get("abnormal_labs"):
            lab_lines = []
            for lab in findings["abnormal_labs"][:12]:
                low = lab.get("reference_low")
                high = lab.get("reference_high")
                reference = ""
                if low is not None and high is not None:
                    reference = f" reference {low:g}-{high:g}"
                elif high is not None:
                    reference = f" desirable <= {high:g}"
                elif low is not None:
                    reference = f" desirable >= {low:g}"
                value = self._format_lab_value(lab.get("value"))
                lab_lines.append(
                    f"- {lab['name']}: {value} {lab.get('unit', '')} "
                    f"({lab['status'].replace('_', ' ')}, {lab.get('severity', 'unknown')} severity;{reference})"
                )
            parts.append("Abnormal or borderline laboratory values:\n" + "\n".join(lab_lines))

        if findings.get("abnormal_labs"):
            suggestion_lines = []
            for lab in findings["abnormal_labs"][:8]:
                suggestions = lab.get("suggestions") or []
                if suggestions:
                    suggestion_lines.append(
                        f"- {lab['name']} target: {lab.get('target', 'within reference range')}. "
                        f"Suggested actions: {' '.join(suggestions[:3])}"
                    )
            if suggestion_lines:
                parts.append("Threshold-based suggestions:\n" + "\n".join(suggestion_lines))

        if findings.get("normal_labs"):
            normal_names = [
                f"{lab['name']} {self._format_lab_value(lab.get('value'))} {lab.get('unit', '')}"
                for lab in findings["normal_labs"][:10]
            ]
            parts.append(f"Selected normal laboratory values: {', '.join(normal_names)}")

        if findings.get("risk_flags"):
            parts.append(f"Risk flags from extraction: {', '.join(findings['risk_flags'])}")

        if findings.get("condition_hints"):
            parts.append(f"Clinical areas to review: {', '.join(findings['condition_hints'])}")

        if findings.get("description"):
            parts.append(f"Visual analysis description: {findings['description']}")

        if findings.get("findings"):
            # CNN-based findings labels
            pathologies = [f["name"] for f in findings["findings"] if f.get("name")]
            if pathologies:
                parts.append(f"Detected pathologies: {', '.join(pathologies)}")

        return (
            "\n".join(parts)
            + "\n\nBased on these extracted findings, generate a cautious medical decision-support report. "
            "Focus on the lab abnormalities and their clinical interpretation. Do not invent symptoms "
            "that are not present in the extracted findings."
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

    async def generate_report(self, context: str, max_length: int = 512) -> str:
        """
        Unified generation method using Google Gemma-2.
        Replaces BioGPT and BioBart for all clinical reasoning tasks.
        """
        return await self._generate_with_gemma(context, max_length)

    async def generate_report_from_context(
        self,
        clinical_findings: Dict[str, Any],
        patient_info: Optional[Dict[str, str]] = None,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate a medical report from clinical context using Gemma-2.
        """
        try:
            logger.info("Generating report from clinical findings...")

            prompt = self._build_report_prompt(clinical_findings, patient_info)
            report_text = await self.generate_report(prompt, max_length)
            
            logger.info("✅ Report generated successfully")
            return {
                "status": "success",
                "full_report": report_text,
                "summary": report_text[:200] + "...", # Simplified summary
                "clinical_findings": clinical_findings,
                "patient_info": patient_info,
                "report_type": "clinical_analysis",
                "generation_model": "Gemma-2-2B-IT",
            }

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _generate_with_gemma(self, context: str, max_length: int = 512) -> str:
        """Generate clinical reasoning using Google Gemma-2-2B (GGUF)."""
        if self.phi_reasoner is None:
            return self._generate_safe_fallback(context)

        try:
            logger.info("Running Gemma-2 GGUF reasoning...")
            
            # Use a more descriptive prompt for Gemma
            response = self.phi_reasoner.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "You write cautious medical decision-support summaries for clinicians. "
                            "Do not claim a definitive diagnosis or treatment plan.\n\n"
                            "If the classifier result is uncertain or low-confidence, say that clearly "
                            "and discuss top visual matches only as possibilities, not likelihoods.\n\n"
                            "Analyze these findings and provide a concise clinical decision-support summary:\n"
                            f"{context}"
                        ),
                    }
                ],
                max_tokens=max_length,
                temperature=0.1
            )
            
            description = response["choices"][0]["message"]["content"]
            return description.strip()

        except Exception as e:
            logger.error(f"Gemma-2 GGUF reasoning failed: {e}")
            return self._generate_safe_fallback(context)

    def _generate_safe_fallback(self, context: str) -> str:
        """Deterministic reasoning fallback when the local LLM is unavailable."""
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
        risk = "urgent" if any(term in lowered for term in urgent_terms) else "routine"
        next_step = (
            "Escalate for urgent clinician review now."
            if risk == "urgent"
            else "Review with a clinician and correlate with symptoms, vitals, and history."
        )

        return (
            "Clinical decision-support summary:\n"
            f"- Input findings: {context[:900]}\n"
            f"- Risk flag: {risk}\n"
            f"- Suggested next step: {next_step}\n"
            "- Safety note: This AI output is not a diagnosis or treatment plan."
        )

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
