"""
PDF extraction utilities
Extract text from medical PDF reports
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.

    Uses PyPDF2 with a fallback to pdfminer.six.

    Args:
        pdf_path: Absolute path to the PDF file

    Returns:
        Extracted text as a single string
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = ""

    # Try PyPDF2 first (fast)
    try:
        import PyPDF2  # type: ignore

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [reader.pages[i].extract_text() or "" for i in range(len(reader.pages))]
            text = "\n".join(pages).strip()

        if text:
            logger.info(f"Extracted {len(text)} chars from {path.name} via PyPDF2")
            return text

    except ImportError:
        logger.debug("PyPDF2 not installed, trying pdfminer")
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")

    # Fallback to pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract  # type: ignore

        text = pdfminer_extract(pdf_path).strip()
        logger.info(f"Extracted {len(text)} chars from {path.name} via pdfminer")
        return text

    except ImportError:
        raise RuntimeError("No PDF extraction library found. Install PyPDF2 or pdfminer.six.")
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")


def clean_extracted_text(text: str) -> str:
    """Remove common PDF extraction artifacts."""
    import re

    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove null bytes
    text = text.replace("\x00", "")
    return text.strip()
