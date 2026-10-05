# DiagnoBot — Multimodal Medical Triage & Clinical Decision Support

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3.2%20%7C%20Moondream-black.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DiagnoBot is a multimodal medical decision-support and triage prototype designed for **urgency assessment, clinical reasoning, and specialist referral recommendations**. It combines **dermatological computer vision & visual observation** with **clinical NLP and symptom analysis** to assist healthcare workers and patients in prioritizing acute medical care.

> ⚠️ **Clinical Disclaimer**: DiagnoBot is an exploratory clinical decision-support prototype and is **not** a certified medical diagnostic device. All outputs, risk scores, image observations, and triage suggestions must be reviewed, verified, and correlated by a qualified healthcare professional.

---

## 🌟 Key Architecture Pillars

DiagnoBot operates on a dual-pillar multimodal framework that bridges visual features with structured clinical intelligence:

```
                      ┌──────────────────────────────────────────────┐
                      │    DiagnoBot: Multimodal Medical Triage      │
                      └──────────────────────┬───────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
                     ▼                                               ▼
         [ Pillar 1: Visual Analysis ]               [ Pillar 2: Clinical Text Analysis ]
         • Skin & Wound Photo / Live Webcam          • Doctor Workspace Clinical Notes
         • MobileNetV3 (10 Disease Classes ~280ms)   • Clinical Extractor (Vitals/Labs/Symptoms)
         • Ollama Moondream (Visual Observation ~2s) • Ollama Llama 3.2 (Clinical Second Opinion)
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                         [ Explainable Triage Decision Engine ]
                         • Urgency Tiers: EMERGENCY | SOON | ROUTINE
                         • Specialist Referral Mapping (Dermatology, ER, etc.)
                         • Red-Flag Extraction & Actionable Next Steps
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
          [ Modern Web Dashboard ]                       [ RESTful FastAPI Backend ]
          • Doctor Workspace                             • /api/v1/dermatology/detect
          • Patient Consultation                         • /api/v1/doctor/second-opinion
          • Emergency Room Monitor                       • /api/v1/triage/emergency
          • Interactive AI Assistant                     • /api/v1/upload/image
```

### 1. Visual Skin & Wound Triage (Pillar 1)
- **Neural Classification Backbone**: Custom fine-tuned `MobileNetV3Small` (`models/skin_mobilenet.pt`, 6.2 MB) evaluating 10 dermatological classes:
  - *Acne*, *Eczema*, *Lupus Erythematosus*, *Impetigo*, *Melanocytic Nevus*, *Bullous Disease*, *Urticaria (Hives)*, *Candidiasis*, *Molluscum Contagiosum*, and *Acanthosis Nigricans*.
- **Vision-Language Observation**: Local **Moondream** vision model (via Ollama) converts raw imagery into objective clinical observations (erythema, border asymmetry, swelling, ulceration, color cast) without hallucinating premature diagnostic claims.
- **Webcam & Live Capture**: Direct camera feed capture for immediate triage screening.

### 2. Doctor Workspace & Clinical Text Analysis (Pillar 2)
- **Clinical Rule Extractor**: High-speed, deterministic parsing of reported symptoms, abnormal laboratory ranges, vital signs (heart rate, blood pressure, SpO2), and temperature thresholds ($<1\text{ms}$).
- **AI Clinical Second Opinion**: Local **Llama 3.2** reasoning model (via Ollama) generates structured clinical decision-support summaries comparing provisional clinical notes against red flags and observed findings.

### 3. Explainable Triage Rules
- Transparent rule engine that maps findings to three actionable tiers:
  - 🚨 **EMERGENCY**: Life-threatening indications, extreme vitals, suspected necrotizing infections, severe fever $\ge 103^\circ\text{F}$, anaphylaxis risk.
  - ⚠️ **SOON**: Moderate symptoms, spreading rash, persistent infection, non-acute systemic signs requiring prompt attention.
  - 🟢 **ROUTINE**: Low-risk superficial lesions and stable conditions suitable for standard outpatient scheduling.

---

## ⚡ 100% Local Inference & Zero Billing

- **No Paid APIs Required**: Powered by local weights (`models/skin_mobilenet.pt`) and local Ollama models (`moondream`, `llama3.2`).
- **Low Hardware Footprint**: Optimized CPU multi-threading, dynamic lazy loading of ML pipelines, and constrained token generation ensure responsive performance on consumer laptops and desktop workstations.

---

## 📂 Project Structure

```text
api/
  middleware/
    error_handlers.py      # Global exception & HTTP error middleware
  routes/
    analytics.py           # Analysis history, metrics & database maintenance
    clinical_workspace.py  # Doctor clinical workspace & emergency triage
    dermatology.py         # Visual skin detection & webcam capture
    health.py              # System health, readiness & model status checks
    nlp_analysis.py        # NLP analysis endpoints
    report_generation.py   # Formal medical report generation
    upload.py              # Unified image and clinical document uploads
    xray_analysis.py       # Chest X-ray analysis endpoints
config/
  logging_config.py        # Structured logging configuration
  settings.py              # Pydantic environment settings (.env)
database/
  connection.py            # Async SQLite connection (SQLAlchemy + aiosqlite)
  crud.py                  # Database operations (sessions, reports, patients)
docker/
  Dockerfile               # Container definition for production deployment
  docker-compose.yml       # Multi-service container orchestration
frontend/
  index.html               # Main portal & navigation gateway
  doctor-dashboard.html    # Clinical notes, lab findings & AI second opinion
  patient-dashboard.html   # Patient symptom intake & skin image upload
  emergency-monitor.html   # Real-time emergency triage & webcam monitor
  ai-assistant.html        # Interactive AI medical consultation chatbot
  report-center.html       # Patient report viewing and export center
ml_pipeline/
  model_manager.py         # Thread-safe model registry & lazy loading
  nlp/
    clinical_extractor.py  # Deterministic symptoms, vitals & labs parser
    report_generator.py    # Llama 3.2 via Ollama clinical reasoning backend
  triage/
    rules.py               # Deterministic urgency & specialist referral rules
  vision/
    derm_cnn.py            # MobileNetV3 PyTorch classifier
    qwen_vl_analyzer.py    # Moondream vision analyzer via Ollama
    xray_analyzer.py       # Chest X-ray feature analyzer
models/
  class_names.json         # 10 dermatological disease label mappings
  database.py              # SQLAlchemy database ORM models
  schemas.py               # Pydantic input/output validation schemas
  skin_mobilenet.pt        # Trained MobileNetV3 PyTorch weights (6.2 MB)
tests/                     # Pytest automated test suite
test_llama.py              # CLI smoke test for Ollama Llama 3.2 reasoning
test_moondream.py          # CLI smoke test for Ollama Moondream vision
main.py                    # FastAPI application entrypoint & static mounting
requirements.txt           # Python package dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+** (Python 3.11+ recommended)
- **Git**
- **Ollama** ([Download Ollama](https://ollama.com/download))

### 2. Pull Ollama Models
Ensure Ollama is running, then pull the local vision and reasoning models:

```bash
# Vision-language model for image observations
ollama pull moondream

# Clinical reasoning model for decision support
ollama pull llama3.2
```

### 3. Clone & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Msadha27/Diagnobot.git
cd Diagnobot

# Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment (Optional)
Copy `.env.example` to `.env` if custom host, port, or logging settings are needed:

```bash
cp .env.example .env
```

### 6. Run the Application

```bash
python main.py
```

The application will start on **`http://127.0.0.1:8000`**.

---

## 🖥️ Interactive Web Dashboards

Once the server is running, explore the built-in dashboards directly in your browser:

| Interface | URL | Description |
| :--- | :--- | :--- |
| **Main Portal** | [http://127.0.0.1:8000/dashboard/](http://127.0.0.1:8000/dashboard/) | Central landing page connecting all modules |
| **Doctor Workspace** | [http://127.0.0.1:8000/dashboard/doctor-dashboard.html](http://127.0.0.1:8000/dashboard/doctor-dashboard.html) | Clinical notes entry, lab values, and AI second opinion |
| **Patient Dashboard** | [http://127.0.0.1:8000/dashboard/patient-dashboard.html](http://127.0.0.1:8000/dashboard/patient-dashboard.html) | Patient consultation, image upload & guidance |
| **Emergency Monitor** | [http://127.0.0.1:8000/dashboard/emergency-monitor.html](http://127.0.0.1:8000/dashboard/emergency-monitor.html) | Acute vital signs triage & live webcam capture |
| **AI Assistant** | [http://127.0.0.1:8000/dashboard/ai-assistant.html](http://127.0.0.1:8000/dashboard/ai-assistant.html) | Interactive medical query & triage assistant |
| **Report Center** | [http://127.0.0.1:8000/dashboard/report-center.html](http://127.0.0.1:8000/dashboard/report-center.html) | Historical reports, triage records & export |
| **Swagger API Docs** | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Interactive OpenAPI documentation & test sandbox |

---

## 📡 Primary API Endpoints

### Dermatology & Vision
- `POST /api/v1/dermatology/detect` — Upload a skin/wound image for neural classification, Moondream visual observation, and triage urgency.
- `POST /api/v1/dermatology/capture` — Capture a frame from an attached webcam for real-time analysis.
- `GET /api/v1/dermatology/classes` — Retrieve the list of supported dermatological disease classes.

### Clinical Workspace & Emergency Triage
- `POST /api/v1/doctor/second-opinion` — Analyze doctor notes, patient symptoms, and vital signs with Llama 3.2 AI clinical decision support.
- `POST /api/v1/triage/emergency` — Fast symptom and vital sign triage with immediate urgency classification (EMERGENCY / SOON / ROUTINE).

### Uploads & Data
- `POST /api/v1/upload/image` — Unified endpoint for image validation and temporary storage.
- `POST /api/v1/upload/document` — Clinical document (PDF / text) intake and extraction.
- `GET /api/v1/history` — Fetch historical triage records and consultation reports.
- `GET /api/v1/health` — System health, device status, and ML model readiness.

---

## 🧪 Testing & Verification

Run the automated test suite with pytest:

```bash
pytest
```

Or verify local model pipelines with standalone verification scripts:

```bash
# Verify Ollama Llama 3.2 reasoning
python test_llama.py

# Verify Ollama Moondream image analysis
python test_moondream.py
```

---

## 📄 License & Attribution

This project is distributed under the MIT License. See [LICENSE](LICENSE) for more details.
Model architectures and weights are subject to their respective open-source licenses (Meta Llama 3.2 Community License, Moondream, and MobileNetV3).
