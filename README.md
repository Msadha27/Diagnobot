# 🩺 DiagnoBot: Multimodal Medical AI Assistant

DiagnoBot is a high-performance, CPU-optimized medical analysis platform designed to assist clinicians with rapid diagnostics using state-of-the-art Vision-Language Models (VLMs) and traditional CNNs.

## 🚀 Core Features

- **X-Ray Analysis**: Chest X-ray pathology detection (DenseNet121) combined with natural language clinical descriptions.
- **Dermatology Detection**: Skin condition classification (HAM10000) with detailed morphological analysis.
- **Clinical Report Generation**: Automated generation of professional medical reports using the Phi-3.5-Mini reasoning engine.
- **Diagnostic History**: Full database persistence to track patient records and analysis trends.
- **Lightweight Architecture**: Optimized to run on standard CPUs with a footprint of less than 4GB RAM.

## 🛠️ Technology Stack

- **Backend**: FastAPI (Async Python 3.10+)
- **Database**: SQLite (SQLAlchemy + aiosqlite)
- **ML Vision**: TorchXRayVision, Moondream2 (1.6B)
- **ML NLP**: Microsoft Phi-3.5-Mini, BioGPT, ClinicalT5
- **Processing**: Optimized for CPU inference using half-precision (float16/bfloat16)

## 📁 Project Structure

- `/api`: FastAPI routes and endpoint logic.
- `/database`: Database connections and CRUD helpers.
- `/ml_pipeline`: Core AI analyzers (Vision, Dermatology, NLP).
- `/models`: Database ORM schemas.
- `/config`: App settings and logging configuration.

## ⚙️ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Server**:
   ```bash
   python main.py
   ```
3. **Access API Docs**:
   Navigate to `http://localhost:8000/docs`

---
*Disclaimer: This is an AI-assisted diagnostic tool. All results must be reviewed by a licensed medical professional.*