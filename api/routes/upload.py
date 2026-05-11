"""
File upload routes – images, PDFs, and plain text
"""

import logging
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from database import crud
from ml_pipeline.nlp.clinical_extractor import extract_clinical_data
from utils.pdf_extraction import clean_extracted_text, extract_text_from_pdf

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/tiff", "image/pjpeg"}
ALLOWED_DOC_TYPES = {"application/pdf", "text/plain", "text/csv"}
MAX_IMAGE_MB = 50
MAX_DOC_MB = 100
IMAGE_ANALYSIS_MODES = {"auto", "skin", "wound", "eye", "fever", "xray"}


def _check_size(content: bytes, max_mb: int, label: str) -> None:
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            status_code=413,
            detail=f"{label} too large. Maximum size: {max_mb} MB (uploaded: {size_mb:.1f} MB)",
        )


@router.post("/upload/image", tags=["upload"])
async def upload_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload an X-ray or dermatology image for analysis."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type '{file.content_type}'. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    content = await file.read()
    _check_size(content, MAX_IMAGE_MB, "Image file")

    # Save with unique name
    suffix = Path(file.filename).suffix or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / unique_name

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    logger.info(f"Image uploaded: {file.filename} → {save_path}")

    return {
        "status": "success",
        "message": "Image uploaded successfully",
        "original_filename": file.filename,
        "saved_as": str(save_path),
        "file_id": unique_name,
        "size_bytes": len(content),
        "content_type": file.content_type,
    }


@router.post("/analyze/upload", tags=["upload", "analysis"])
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    analysis_mode: str = Form("auto"),
    patient_id: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    symptoms: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Unified upload pipeline.

    PDFs/text are extracted and sent through the clinical NLP/report pipeline.
    Images are routed to X-ray, skin, wound, eye, or fever visual analysis.
    """
    logger.info(
        "Unified upload received: filename=%s content_type=%s mode=%s patient_id=%s",
        file.filename,
        file.content_type,
        analysis_mode,
        patient_id,
    )

    mode = analysis_mode.lower().strip()
    if mode not in IMAGE_ANALYSIS_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid analysis_mode. Use one of: {', '.join(sorted(IMAGE_ANALYSIS_MODES))}",
        )

    content = await file.read()
    if file.content_type in ALLOWED_DOC_TYPES:
        _check_size(content, MAX_DOC_MB, "Document file")
        return await _process_document_upload(file, content, patient_id, db)

    if file.content_type in ALLOWED_IMAGE_TYPES or _looks_like_image(file.filename):
        _check_size(content, MAX_IMAGE_MB, "Image file")
        return await _process_image_upload(
            file=file,
            content=content,
            mode=mode,
            patient_id=patient_id,
            temperature=temperature,
            symptoms=symptoms,
            db=db,
        )

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported file type '{file.content_type}'. Upload a PDF/text report "
            "or a JPEG/PNG/TIFF image."
        ),
    )


@router.post("/upload/pdf", tags=["upload"])
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a medical PDF report for text extraction."""
    if file.content_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type. Allowed: {', '.join(ALLOWED_DOC_TYPES)}",
        )

    content = await file.read()
    _check_size(content, MAX_DOC_MB, "Document file")

    suffix = Path(file.filename).suffix or ".pdf"
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / unique_name

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    logger.info(f"Document uploaded: {file.filename} → {save_path}")

    return {
        "status": "success",
        "message": "Document uploaded successfully",
        "original_filename": file.filename,
        "saved_as": str(save_path),
        "file_id": unique_name,
        "size_bytes": len(content),
    }


@router.post("/upload/text", tags=["upload"])
async def upload_text(text: str = Form(...), patient_id: Optional[str] = Form(None)) -> Dict[str, Any]:
    """Submit patient symptoms or notes as plain text."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")

    text_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{text_id}.txt"

    async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
        await f.write(text)

    logger.info(f"Text input saved: {save_path}")

    return {
        "status": "success",
        "message": "Text submitted successfully",
        "text_id": text_id,
        "saved_as": str(save_path),
        "patient_id": patient_id,
        "word_count": len(text.split()),
    }


async def _save_upload(file: UploadFile, content: bytes, file_type: str, db: AsyncSession) -> Path:
    suffix = Path(file.filename or "").suffix or _default_suffix(file.content_type)
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / unique_name

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    await crud.save_upload_record(
        db,
        file_id=unique_name,
        original_filename=file.filename or unique_name,
        saved_path=str(save_path),
        file_type=file_type,
        size_bytes=len(content),
        content_type=file.content_type or "application/octet-stream",
    )
    return save_path


async def _process_document_upload(
    file: UploadFile,
    content: bytes,
    patient_id: Optional[str],
    db: AsyncSession,
) -> Dict[str, Any]:
    save_path = await _save_upload(file, content, "document", db)
    record = await crud.create_analysis_record(
        db,
        analysis_type="document_nlp",
        patient_id=patient_id,
        input_file=file.filename,
    )

    try:
        if file.content_type == "application/pdf":
            extracted_text = clean_extracted_text(extract_text_from_pdf(str(save_path)))
        else:
            extracted_text = content.decode("utf-8", errors="ignore").strip()

        if not extracted_text:
            raise ValueError("No readable text was extracted from the uploaded document.")

        extraction = extract_clinical_data(extracted_text)
        report = await _generate_report(
            clinical_findings={
                "clinical_summary": extraction["summary"],
                "symptoms": extraction["symptoms"],
                "lab_values": extraction["lab_values"],
                "lab_results": extraction["lab_results"],
                "abnormal_labs": extraction["abnormal_labs"],
                "normal_labs": extraction["normal_labs"][:12],
                "report_kind": extraction["report_kind"],
                "condition_hints": extraction["condition_hints"],
                "risk_flags": extraction["risk_flags"],
            },
            patient_id=patient_id,
            db=db,
        )

        result = {
            "status": "success",
            "pipeline": "document -> text extraction -> NLP extraction -> report summary",
            "file": {
                "original_filename": file.filename,
                "saved_as": str(save_path),
                "content_type": file.content_type,
                "size_bytes": len(content),
            },
            "extracted_text_preview": extracted_text[:800],
            "nlp": _compact_extraction(extraction),
            "report": _compact_report(report),
        }
        if record is not None:
            await crud.update_analysis_result(
                db,
                record.id,
                result,
                status="success",
                model_used="PDF/Text extractor + clinical extractor + Gemma fallback",
                confidence=1.0 if extraction["symptoms"] or extraction["lab_values"] else 0.5,
            )
        return result
    except Exception as exc:
        if record is not None:
            await crud.update_analysis_result(db, record.id, {"error": str(exc)}, status="error")
        raise HTTPException(status_code=500, detail=str(exc))


async def _process_image_upload(
    file: UploadFile,
    content: bytes,
    mode: str,
    patient_id: Optional[str],
    temperature: Optional[float],
    symptoms: Optional[str],
    db: AsyncSession,
) -> Dict[str, Any]:
    save_path = await _save_upload(file, content, "image", db)
    selected_mode = _infer_image_mode(file.filename or "", mode)

    if selected_mode == "xray":
        return await _run_xray_pipeline(save_path, file, patient_id, db)
    return await _run_visual_symptom_pipeline(
        save_path=save_path,
        file=file,
        mode=selected_mode,
        patient_id=patient_id,
        temperature=temperature,
        symptoms=symptoms,
        db=db,
    )


def _compact_extraction(extraction: Dict[str, Any]) -> Dict[str, Any]:
    abnormal_labs = extraction.get("abnormal_labs", [])
    normal_labs = extraction.get("normal_labs", [])

    return {
        "status": extraction.get("status"),
        "report_kind": extraction.get("report_kind"),
        "summary": extraction.get("summary"),
        "important_findings": [_compact_lab(lab, include_suggestions=True) for lab in abnormal_labs[:8]],
        "normal_baseline": [_compact_lab(lab, include_suggestions=False) for lab in normal_labs[:10]],
        "lab_values": extraction.get("lab_values", {}),
        "condition_hints": extraction.get("condition_hints", []),
        "risk_flags": extraction.get("risk_flags", []),
        "counts": {
            "total_labs": len(extraction.get("lab_results", [])),
            "abnormal_labs": len(abnormal_labs),
            "normal_labs": len(normal_labs),
        },
        "disclaimer": extraction.get("disclaimer"),
    }


def _compact_lab(lab: Dict[str, Any], include_suggestions: bool) -> Dict[str, Any]:
    compact = {
        "name": lab.get("name"),
        "value": lab.get("value"),
        "unit": lab.get("unit"),
        "reference_range": _reference_range(lab),
        "status": lab.get("status"),
        "severity": lab.get("severity"),
        "target": lab.get("target"),
    }
    if include_suggestions:
        compact["interpretation"] = lab.get("interpretation")
        compact["suggestions"] = lab.get("suggestions", [])[:3]
    return compact


def _reference_range(lab: Dict[str, Any]) -> str:
    low = lab.get("reference_low")
    high = lab.get("reference_high")
    unit = lab.get("unit", "")
    if low is not None and high is not None:
        return f"{low:g}-{high:g} {unit}".strip()
    if high is not None:
        return f"<= {high:g} {unit}".strip()
    if low is not None:
        return f">= {low:g} {unit}".strip()
    return "not configured"


def _compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    full_report = report.get("full_report", "")
    return {
        "status": report.get("status"),
        "summary": report.get("summary"),
        "full_report_preview": full_report[:1200] + ("..." if len(full_report) > 1200 else ""),
        "generation_model": report.get("generation_model"),
    }


async def _run_xray_pipeline(
    save_path: Path,
    file: UploadFile,
    patient_id: Optional[str],
    db: AsyncSession,
) -> Dict[str, Any]:
    from api.routes.xray_analysis import get_xray_analyzers

    record = await crud.create_analysis_record(db, "xray", patient_id, file.filename)
    try:
        analyzer, vision_analyzer = await get_xray_analyzers()
        result = await analyzer.analyze_xray(str(save_path))
        vision_result = await vision_analyzer.analyze_xray(str(save_path))
        if vision_result.get("status") == "success":
            result["description"] = vision_result.get("description")
            result["vlm_model"] = vision_result.get("model")
        await _attach_doctor_verdict(result, patient_id, db)
        if record is not None:
            await crud.update_analysis_result(
                db,
                record.id,
                result,
                status="success",
                model_used=f"TorchXRayVision + {result.get('vlm_model', 'vision model')}",
                confidence=_first_confidence(result),
            )
        return {"status": "success", "pipeline": "image -> xray model -> report reasoning", "analysis": result}
    except Exception as exc:
        if record is not None:
            await crud.update_analysis_result(db, record.id, {"error": str(exc)}, status="error")
        raise HTTPException(status_code=500, detail=str(exc))


async def _run_visual_symptom_pipeline(
    save_path: Path,
    file: UploadFile,
    mode: str,
    patient_id: Optional[str],
    temperature: Optional[float],
    symptoms: Optional[str],
    db: AsyncSession,
) -> Dict[str, Any]:
    from api.routes.dermatology import get_derm_analyzers

    record = await crud.create_analysis_record(db, f"visual_{mode}", patient_id, file.filename)
    try:
        classifier, vision_analyzer = await get_derm_analyzers()
        extra_context = _visual_context(mode, temperature, symptoms)

        classification = None
        if mode == "skin" and classifier is not None:
            classification = await classifier.analyze_skin_image(str(save_path))

        if mode == "wound":
            vision_result = await vision_analyzer.analyze_wound(str(save_path), extra_context)
        elif mode == "eye":
            vision_result = await vision_analyzer.analyze_eye(str(save_path), extra_context)
        elif mode == "fever":
            vision_result = await vision_analyzer.analyze_fever(str(save_path), extra_context)
        else:
            vision_result = await vision_analyzer.analyze_skin(str(save_path), extra_context)

        result = {
            "status": "success",
            "analysis_type": mode,
            "classification": classification.get("classification") if classification else None,
            "all_predictions": classification.get("all_predictions") if classification else None,
            "clinical_advice": classification.get("clinical_advice") if classification else None,
            "description": vision_result.get("description"),
            "vlm_model": vision_result.get("model"),
            "symptoms": symptoms,
            "temperature": temperature,
            "chart_data": _chart_data(classification, mode, vision_result),
            "disclaimer": "AI decision support only. This is not a diagnosis.",
        }
        result = {key: value for key, value in result.items() if value is not None}
        await _attach_doctor_verdict(result, patient_id, db)
        if record is not None:
            await crud.update_analysis_result(
                db,
                record.id,
                result,
                status="success",
                model_used=f"{mode} visual analyzer + report reasoning",
                confidence=_visual_confidence(result),
            )
        return {"status": "success", "pipeline": f"image -> {mode} visual model -> report reasoning", "analysis": result}
    except Exception as exc:
        if record is not None:
            await crud.update_analysis_result(db, record.id, {"error": str(exc)}, status="error")
        raise HTTPException(status_code=500, detail=str(exc))


async def _attach_doctor_verdict(result: Dict[str, Any], patient_id: Optional[str], db: AsyncSession) -> None:
    report = await _generate_report(result, patient_id, db)
    result["doctor_verdict"] = report.get("summary") or report.get("full_report")
    result["reasoning_model"] = report.get("generation_model", "report generator")


async def _generate_report(
    clinical_findings: Dict[str, Any],
    patient_id: Optional[str],
    db: AsyncSession,
) -> Dict[str, Any]:
    try:
        from main import model_manager
        from ml_pipeline.nlp.report_generator import create_report_generator

        generator = await create_report_generator(model_manager)
        report = await generator.generate_report_from_context(
            clinical_findings=clinical_findings,
            patient_info={"id": patient_id} if patient_id else None,
        )
        if report.get("status") == "success":
            await crud.save_medical_report(
                db,
                full_report=report["full_report"],
                report_type="automated_upload_analysis",
                generation_model=report.get("generation_model", "Gemma fallback"),
                summary=report.get("summary"),
                patient_id=patient_id,
                clinical_findings=clinical_findings,
            )
        return report
    except Exception as exc:
        logger.warning("Report generation fallback used: %s", exc)
        summary = clinical_findings.get("clinical_summary") or clinical_findings.get("description") or str(clinical_findings)[:500]
        return {
            "status": "success",
            "summary": f"Clinical decision-support summary: {summary}",
            "generation_model": "local fallback",
        }


def _infer_image_mode(filename: str, mode: str) -> str:
    if mode != "auto":
        return mode
    lowered = filename.lower()
    if "xray" in lowered or "x-ray" in lowered or "chest" in lowered:
        return "xray"
    if "wound" in lowered or "ulcer" in lowered or "cut" in lowered:
        return "wound"
    if "eye" in lowered or "sclera" in lowered:
        return "eye"
    if "fever" in lowered or "face" in lowered:
        return "fever"
    return "skin"


def _quick_doctor_verdict(result: Dict[str, Any]) -> str:
    analysis_type = result.get("analysis_type", "image")
    classification = result.get("classification") or {}
    description = result.get("description") or "No visual description was returned."
    symptoms = result.get("symptoms")
    temperature = result.get("temperature")

    label = (
        classification.get("disease")
        or classification.get("label")
        or classification.get("top_match")
        or "not classified"
    )
    confidence = classification.get("confidence")

    risk_words = ["urgent", "severe", "bleeding", "pus", "discharge", "necrosis", "yellow", "high fever"]
    lowered = f"{description} {symptoms or ''}".lower()
    risk_flag = "urgent review suggested" if any(word in lowered for word in risk_words) else "routine review suggested"

    confidence_text = ""
    if isinstance(confidence, (int, float)):
        confidence_text = f" Confidence: {confidence:.0%}."

    context = []
    if temperature is not None:
        context.append(f"temperature reported: {temperature}")
    if symptoms:
        context.append(f"symptoms: {symptoms}")
    context_text = f" Context: {', '.join(context)}." if context else ""

    return (
        f"Clinical decision-support summary: {analysis_type} image analyzed. "
        f"Classifier result: {label}.{confidence_text} "
        f"Visual observation: {description[:700]} "
        f"{context_text} Risk flag: {risk_flag}. "
        "This is not a diagnosis; correlate with patient history and clinician review."
    )


def _looks_like_image(filename: Optional[str]) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in {".jpg", ".jpeg", ".jfif", ".png", ".tif", ".tiff"}


def _visual_context(mode: str, temperature: Optional[float], symptoms: Optional[str]) -> str:
    parts = [f"Requested visual symptom mode: {mode}."]
    if temperature is not None:
        parts.append(f"Measured/reported temperature: {temperature}.")
    if symptoms:
        parts.append(f"Reported symptoms: {symptoms}.")
    return " ".join(parts)


def _chart_data(classification: Optional[Dict[str, Any]], mode: str, vision_result: Dict[str, Any]) -> Dict[str, Any]:
    if classification and classification.get("all_predictions"):
        return {
            "type": "prediction_probabilities",
            "items": [
                {
                    "label": item.get("disease") or item.get("label") or item.get("name", "Unknown"),
                    "value": round(float(item.get("confidence", 0)) * 100, 1),
                }
                for item in classification["all_predictions"][:5]
            ],
        }
    description = (vision_result.get("description") or "").lower()
    signals = {
        "redness": 80 if "red" in description else 30,
        "swelling": 75 if "swelling" in description else 25,
        "discharge": 70 if "discharge" in description or "pus" in description else 20,
        "urgent flag": 90 if "urgent" in description else 25,
    }
    if mode == "eye":
        signals = {
            "redness": 80 if "red" in description else 25,
            "yellowing": 80 if "yellow" in description else 20,
            "pallor": 70 if "pale" in description or "pallor" in description else 20,
            "swelling": 70 if "swelling" in description else 20,
        }
    if mode == "fever":
        signals = {
            "flushed": 75 if "flush" in description or "red" in description else 25,
            "sweating": 75 if "sweat" in description else 20,
            "fatigue": 70 if "fatigue" in description or "tired" in description else 25,
            "rash": 80 if "rash" in description else 20,
        }
    return {"type": "visual_signals", "items": [{"label": key, "value": value} for key, value in signals.items()]}


def _first_confidence(result: Dict[str, Any]) -> float:
    findings = result.get("findings") or []
    if findings:
        return float(findings[0].get("confidence", 0.0))
    return 0.0


def _visual_confidence(result: Dict[str, Any]) -> float:
    classification = result.get("classification") or {}
    return float(classification.get("confidence", 0.65 if result.get("description") else 0.0))


def _default_suffix(content_type: Optional[str]) -> str:
    return {
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "image/png": ".png",
        "image/tiff": ".tiff",
    }.get(content_type or "", ".bin")
