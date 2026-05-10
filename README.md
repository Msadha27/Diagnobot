# DiagnoBot

DiagnoBot is a FastAPI backend for medical decision-support demos. It is not a diagnostic device and all outputs must be reviewed by a qualified clinician.

## Core Features

1. Emergency webcam capture for skin/wound screening.
2. Skin or wound image analysis using Moondream GGUF by default.
3. X-ray upload analysis using TorchXRayVision plus vision-language description.
4. Medical report text/PDF workflows and generated clinician-style summaries.
5. SQLite storage for analysis records, uploads, and generated reports.

## Current Model Stack

- Vision: Moondream2 GGUF, configured by `VISION_MODEL_BACKEND=moondream_gguf`.
- Future vision option: PaliGemma, configured by `VISION_MODEL_BACKEND=paligemma` on a stronger machine.
- X-ray labels: TorchXRayVision DenseNet.
- Reasoning: Gemma GGUF when available, with a safe deterministic fallback.
- Database: SQLite via SQLAlchemy async.

## Project Structure

```text
api/                 FastAPI routes, dependencies, middleware
config/              App settings and logging
database/            Async database connection and CRUD helpers
models/              SQLAlchemy and request/response schemas
ml_pipeline/         Vision, X-ray, NLP, and model manager code
utils/               PDF extraction, validation, image helpers
tests/               API and pipeline tests
docker/              Docker files
data/                Local skin image dataset
uploads/             Runtime uploaded files, ignored by git
logs/                Runtime logs, ignored by git
models_cache/        Downloaded model cache, ignored by git
archive/             Old prototypes kept out of the main app path
```

## Run Locally

```powershell
cd "C:\Users\HP\Documents\Ding Dong bot"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Useful Endpoints

- `POST /api/v1/dermatology/detect`
- `POST /api/v1/dermatology/capture`
- `POST /api/v1/xray/analyze`
- `POST /api/v1/report/generate`
- `POST /api/v1/nlp/analyze-text`

## Notes

The old YOLO prototype files were moved to `archive/legacy_yolo_demo`. The main runnable backend is `main.py`.
