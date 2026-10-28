# DiagnoBot

DiagnoBot is a prototype for image-assisted health triage and clinical note support. It combines image classifiers, local language/vision models, deterministic symptom extraction, and urgency rules behind a FastAPI service and browser dashboards.

> **Research/demo prototype only.** It is not a diagnostic device and must not be used to make treatment decisions. Outputs require review by a qualified clinician.

## Implemented components

- **Skin images:** locally stored, fine-tuned PyTorch **MobileNetV3-Small** predicts one of ten folders in `data/skin_diseases`. See [MODEL_CARD.md](MODEL_CARD.md) for the current evaluation.
- **Chest X-rays:** **TorchXRayVision DenseNet-121**, weights `densenet121-res224-mimic_ch`. This is separate from the skin model and was not evaluated in the DiagnoBot skin experiment.
- **Image descriptions:** local **Moondream** served by Ollama. It describes image content; it is not the skin classifier.
- **Clinical text summaries:** local **Llama 3.2** served by Ollama. It summarizes supplied context; it is not a validated diagnostic model.
- **Triage and text extraction:** deterministic Python rules, not learned models or LLMs.
- **Web application:** FastAPI endpoints and browser dashboards for uploads, reports, and triage.

### Model roles and rationale

| Component | Model / implementation | Role | Reason for use |
|---|---|---|---|
| Skin classifier | MobileNetV3-Small, PyTorch, ImageNet-pretrained then fine-tuned | Ten-class image prediction | Compact for CPU/edge experiments; transfer learning is practical with a small dataset. |
| X-ray classifier | TorchXRayVision DenseNet-121 (`densenet121-res224-mimic_ch`) | Multi-label chest X-ray prediction | Existing chest-imaging model; independent from skin classification. |
| Image description | `moondream` via Ollama | Natural-language image observations | Local vision-language inference without a hosted API; observations are unverified. |
| Clinical summary | `llama3.2` via Ollama | Short text summary from supplied context | Local text generation; outputs are not clinically validated. |
| Triage/extraction | Project Python rules | Extract terms and assign urgency cues | Deterministic and inspectable; does not rely on generated text. |

Ollama identifiers are tags (`moondream`, `llama3.2`), not immutable versions. For a reproducible report, record `ollama list` and `ollama show <model>` and preserve the model digests. Older/optional Hugging Face and GGUF loaders also remain in the codebase; they are not the primary models above.

## Current skin-model result

The latest run used 932 images from ten classes, split by image into 652 training, 140 validation, and 140 internal-test images. Test results: **70.7% accuracy**, **70.3% balanced accuracy**, and **0.699 macro-F1**. These are preliminary image-classification results. They do not measure LLM answers, triage, or the complete application, and they are not external clinical validation. Patient/source identifiers were unavailable, so related images may cross partitions. See [metrics](evaluation/skin_test_metrics.json), [confusion matrix](evaluation/skin_test_confusion_matrix.csv), [training history](evaluation/skin_training_history.csv), and [model card](MODEL_CARD.md).

## Hardware status

This repository contains a web backend and model pipelines. It does **not** contain Raspberry Pi GPIO, ESP32 serial, motor-driver, temperature-sensor, pulse-oximeter, or robot-control integration. The Pi, ESP32, and chassis are not controlled by this software. Hardware integration remains a separate task; do not connect triage output to autonomous treatment or movement.

## Setup

Requirements: Python 3.10+, Git, and [Ollama](https://ollama.com/download) for model-backed inference.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

The application defaults to simulation mode (`MOCK_MODE=true`). To use the local model-backed paths, start Ollama and pull the model tags:

```powershell
ollama pull moondream
ollama pull llama3.2
```

Set `MOCK_MODE=false` in `.env`. The skin classifier loads `models/skin_mobilenet.pt`; the X-ray model may download pretrained weights on first use. Memory use and latency depend on hardware.

Start the backend from the repository root:

```powershell
python main.py
```

API docs: `http://127.0.0.1:8000/docs`. The frontend is served under `/dashboard/`.

## Retrain the skin model

The training script uses local folder-per-class images, ImageNet-pretrained MobileNetV3-Small, separate training/evaluation transforms, stratified 70/15/15 image splits, validation macro-F1 for checkpoint selection, early stopping, and a final held-out test evaluation.

```powershell
python train_skin_mobilenet.py
```

It writes `models/skin_mobilenet_candidate.pt` and evaluation artifacts to `evaluation/`. Review results before promoting the candidate to the active checkpoint. Do not tune on the test split; doing so makes it model-selection data.

## Main project files

```text
api/                               FastAPI endpoints
config/settings.py                 Runtime configuration; simulation is on by default
data/skin_diseases/                Folder-per-class skin dataset
evaluation/                        Training history, metrics, confusion matrix
frontend/                          Browser interfaces
ml_pipeline/                       Vision, report, extraction, and triage code
models/skin_mobilenet.pt            Active ten-class skin checkpoint
models/skin_mobilenet_previous.pt   Previous checkpoint backup
train_skin_mobilenet.py             Skin-model training/evaluation script
MODEL_CARD.md                       Skin model results and improvement plan
tests/                              Project automated test suite
main.py                             FastAPI application entry point
requirements.txt                    Python dependencies
```

## Research and safety

Before making clinical claims, obtain governed, representative data with reliable reference labels; split by patient (and ideally source/site); compare relevant baselines; report class-level and calibration metrics with uncertainty; and test on an untouched external cohort. A single accuracy value is insufficient. See the [CLAIM 2024 medical-imaging reporting checklist](https://pubs.rsna.org/doi/10.1148/ryai.240300).
