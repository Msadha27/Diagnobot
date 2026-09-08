# DiagnoBot — Multimodal Medical Triage Assistant

DiagnoBot is a multimodal medical decision-support prototype designed for **urgency assessment and specialist referral recommendations**. It combines **visual skin/wound observation** with **clinical text and symptom analysis** to help prioritize patient care.

> ⚠️ **Clinical Disclaimer**: DiagnoBot is an exploratory clinical decision-support prototype and is **not** a certified diagnostic device. All outputs, risk scores, and recommendations must be reviewed and correlated by a qualified healthcare professional.

---

## 🌟 Key Architecture Pillars

DiagnoBot operates on a dual-pillar multimodal framework:

```
                  ┌──────────────────────────────────────────────┐
                  │    DiagnoBot: Multimodal Medical Triage      │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
     [ Pillar 1: Visual Analysis ]               [ Pillar 2: Clinical Text Analysis ]
     • Skin & Wound Photo / Live Webcam          • Doctor Workspace Note Analysis
     • MobileNetV3Small (Disease hint ~280ms)    • Clinical Extractor (Vitals/Labs/Symptoms <1ms)
     • Moondream2 GGUF (Visual signs ~2s)        • Gemma-2-2B GGUF (AI Second Opinion ~2s)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                     [ Explainable Triage Decision Engine ]
                     • Urgency Tiers: EMERGENCY | SOON | ROUTINE
                     • Specialist Referral Mapping (Dermatology, Emergency, etc.)
                     • Red Flag Identification & Actionable Next Steps
```

### 1. Visual Skin & Wound Triage (Pillar 1)
- **Classification Backbone**: Custom-trained `MobileNetV3Small` (`models/skin_mobilenet.pt`, 6.2 MB) evaluating 10 dermatological disease classes (`Acne`, `Eczema`, `Lupus`, `Impetigo`, `Nevus`, `Bullous`, `Urticaria Hives`, `Candidiasis`, `Molluscum`, `Acanthosis Nigricans`).
- **Vision-Language Observation**: `Moondream2 GGUF` converts raw imagery into structured visible observations (erythema, swelling, border asymmetry, discharge, ulceration) without forcing premature diagnosis.
- **Webcam Integration**: Real-time snapshot screening for emergency triage.

### 2. Doctor Workspace & Clinical Text Analysis (Pillar 2)
- **Clinical Rule Extractor**: Deterministic, transparent parsing of reported symptoms, abnormal lab ranges, vital signs, and temperature thresholds with sub-millisecond execution.
- **AI Clinical Second Opinion**: `Google Gemma-2-2B-IT GGUF` generates concise, cautious decision-support summaries reviewing the clinician's provisional assessment against observed red flags.

### 3. Explainable Triage Rules
- Transparent logic engine mapping signals to **EMERGENCY** (life-threatening signs, severe fever $\ge 103^\circ\text{F}$, necrosis), **SOON** (moderate severity, persistent pain, spreading rash), or **ROUTINE** (low-severity lesions without acute flags).

---

## ⚡ Hardware Efficiency & Zero-Billing

- **100% Local & Offline**: Runs entirely on local weights (`models/skin_mobilenet.pt`, `Moondream2 GGUF`, `Gemma-2-2B GGUF`). Zero external API keys or paid billing subscriptions are required.
- **Sub-3s Latency**: Optimized CPU multi-threading (`CPU_THREADS=4`), image downsampling (384px thumbnail), and constrained token generation (`REASONING_MAX_TOKENS=96`) allow the entire multimodal pipeline to execute smoothly on standard consumer hardware.

---

## 📂 Project Structure

```text
api/
  routes/
    dermatology.py         # Visual skin/wound detection & webcam capture
    clinical_workspace.py  # Doctor workspace & emergency triage
    upload.py              # Unified image/document upload handler
    analytics.py           # Analysis history & database maintenance
    health.py              # System health & model status
config/                    # Pydantic settings & logging configuration
database/                  # Async SQLite database (SQLAlchemy + aiosqlite)
data/                      # Local skin disease dataset (10 categories)
frontend/                  # Responsive web dashboard UI & AI assistant
ml_pipeline/
  vision/
    derm_cnn.py            # MobileNetV3 PyTorch classification
    qwen_vl_analyzer.py    # Moondream2 GGUF visual observation layer
  nlp/
    clinical_extractor.py  # Transparent symptom, vitals & lab parser
    report_generator.py    # Gemma-2-2B GGUF clinical reasoning
  triage/
    rules.py               # Explainable urgency & referral rules
models/
  skin_mobilenet.pt        # Trained MobileNetV3 PyTorch model weights (6.2 MB)
  class_names.json         # Disease class labels
main.py                    # FastAPI application entrypoint
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Virtual environment (`venv`) with project requirements installed

### Quickstart

1. **Activate Virtual Environment**:
   ```powershell
   cd "C:\Users\HP\Documents\Ding Dong bot"
   .\venv\Scripts\Activate.ps1
   ```

2. **Run Backend Server**:
   ```powershell
   python main.py
   ```

3. **Access Interfaces**:
   - **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - **Web Dashboard**: [http://127.0.0.1:8000/dashboard/](http://127.0.0.1:8000/dashboard/)
   - **AI Consultation Center**: [http://127.0.0.1:8000/dashboard/ai-assistant.html](http://127.0.0.1:8000/dashboard/ai-assistant.html)

---

## 📡 Primary API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/dermatology/detect` | Upload a skin image for neural classification, visual observation & triage. |
| `POST` | `/api/v1/dermatology/capture` | Capture and analyze a live frame from the connected webcam. |
| `POST` | `/api/v1/doctor/second-opinion` | Clinical text analysis of findings & plan with AI second opinion. |
| `POST` | `/api/v1/triage/emergency` | Symptom & vital sign triage with immediate urgency rating. |
| `GET` | `/api/v1/history` | Retrieve previous analysis records and reports. |
| `GET` | `/api/v1/health` | Service uptime and model readiness check. |
