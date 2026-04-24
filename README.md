<div align="center">

# 🏥 DiagnoBot

**AI-powered medical diagnosis system with vision + NLP models**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Overview

**DiagnoBot** is a comprehensive backend system designed for medical image analysis and clinical text processing. It integrates state-of-the-art pre-trained models to assist in diagnostic and clinical workflows.

### 🧠 Integrated Models

| Category | Model | Description | Key Features |
| :--- | :--- | :--- | :--- |
| **Vision** | **Moondream2** | X-ray anomaly detection | 1.86B params, zero-shot detection |
| **Vision** | **Derm CNN** | Skin lesion classification | Trained on HAM10000, webcam support |
| **Vision** | **TorchXRayVision** | Multi-dataset X-ray extraction | DenseNet121 architecture |
| **NLP** | **Bio_ClinicalBERT**| Clinical text understanding | Trained on MIMIC-III dataset |
| **NLP** | **BioGPT** | Medical report generation | Autoregressive text generation |
| **NLP** | **BioBart** | Patient input conversion | Converts raw info to formal reports |
| **NLP** | **ClinicalT5** | Medical report summarization | Summarizes long clinical texts |

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python:** 3.10+
- **Hardware:** PyTorch with CUDA support (optional but highly recommended for faster inference)
- **Deployment:** Docker & Docker Compose (for containerized deployment)

### 2. Installation

```bash
# Clone repository
git clone https://github.com/Msadha27/Diagnobot.git
cd Diagnobot

# Create and activate virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your specific settings
```

### 3. Running Locally

**Option A: Direct Python Execution**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Option B: Using Docker Compose**
```bash
docker-compose up --build
```

### 4. Access API Interfaces
- **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 📚 API Endpoints

### 🩺 System & Health
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Basic system health check |
| `GET` | `/api/v1/system/info` | Detailed system and configuration info |

### 🩻 X-Ray Analysis
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/xray/analyze` | Upload chest X-ray. Returns findings, anomalies, scores, advice |
| `POST` | `/api/v1/xray/batch-analyze` | Batch analyze up to 10 X-rays simultaneously |
| `GET`  | `/api/v1/xray/models` | Retrieve model info and active capabilities |

### 🔬 Dermatology (Skin Analysis)
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/dermatology/detect` | Upload skin image. Returns classification, severity, advice |
| `POST` | `/api/v1/dermatology/capture`| Analyze image captured directly from webcam |
| `GET`  | `/api/v1/dermatology/camera` | Start/stop webcam stream interface |

### 📝 Report Generation
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/report/generate` | Generate full report and summary from findings dict |
| `POST` | `/api/v1/report/from-input`| Convert raw patient symptoms/history into formal report |
| `POST` | `/api/v1/report/summarize` | Summarize long and complex medical reports |

### 💬 NLP Analysis
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/nlp/analyze-text` | Analyze clinical text for context, entities, and insights |
| `POST` | `/api/v1/nlp/understand` | Extract focused medical context from general symptoms |
| `POST` | `/api/v1/nlp/extract-entities`| Identify medical Named Entities (drugs, conditions, etc.) |

---

## 📖 Model Documentation

- **Moondream2:** [HuggingFace](https://huggingface.co/vikhyatk/moondream2) • 1.86B params • Fast zero-shot detection.
- **Bio_ClinicalBERT:** [HuggingFace](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) • 110M params • MLM on clinical text.
- **BioGPT:** [GitHub](https://github.com/microsoft/BioGPT) • 360M params • Medical text generation.
- **TorchXRayVision:** [GitHub](https://github.com/mlmed/torchxrayvision) • Specialized X-ray architectures.

---

## 📊 Project Structure

```text
diagnobot/
├── api/
│   ├── main.py                 # FastAPI application root
│   └── routes/                 # Modular endpoint handlers
├── ml_pipeline/
│   ├── model_manager.py        # Model loading & caching logic
│   ├── vision/                 # Moondream2, Derm CNN, TorchXRayVision
│   └── nlp/                    # ClinicalBERT, BioGPT, BioBart
├── config/                     # Configuration and environment settings
├── tests/                      # Unit & integration tests
├── notebooks/                  # Educational Jupyter notebooks
├── docker-compose.yml          # Container configuration
└── requirements.txt            # Python dependencies
```

---

## 💻 Development Commands

**Running Tests**
```bash
pytest tests/ -v --cov=api --cov=ml_pipeline
```

**Code Quality Checks**
```bash
black .                      # Format code
flake8 .                     # Linting
mypy api/ ml_pipeline/       # Type checking
```

---

## 🐛 Troubleshooting

* **GPU Memory Issues:** Reduce `BATCH_SIZE` in settings (e.g., to 1). Enable `USE_QUANTIZATION = true`.
* **Slow Model Download:** Pre-download manually using the `transformers` library script.
* **Webcam Not Detected:** Verify your camera index (`python -c "import cv2; print(cv2.getCameraIndex())"`).

---

## 🎯 Interview Preparation Highlights
* **Moondream2 vs GPT-4V:** Moondream2 is highly localized, zero-cost per image, and ensures total patient data privacy compared to OpenAI models.
* **Zero-shot detection:** Combines CLIP architectures with language prompting to identify medical concepts without custom fine-tuning.
* **Transfer Learning:** Applying general ResNet/MobileNet knowledge specifically to the HAM10000 dataset for skin lesion classification.

---

<div align="center">

**Built with ❤️ for medical AI learning and deployment** <br>
*MIT License - See the [LICENSE](LICENSE) file for more information.*

</div>