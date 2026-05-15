# Running DiagnoBot On A 16GB Laptop

Use the 16GB laptop as the main demo machine for Moondream/Gemma inference.

## 1. Required Software

Install these first:

- Git
- Python 3.10, 64-bit
- Microsoft Visual C++ Build Tools, if `llama-cpp-python` fails during install
- A stable internet connection for the first model download

## 2. Clone The Project

```powershell
cd "C:\Users\<TEAMMATE_NAME>\Documents"
git clone https://github.com/Msadha27/Diagnobot.git
cd Diagnobot
```

If the project was already cloned:

```powershell
cd "C:\Users\<TEAMMATE_NAME>\Documents\Diagnobot"
git pull origin main
```

## 3. Create And Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

## 4. Install Requirements

```powershell
pip install -r requirements.txt
```

This can take time. The heaviest packages are PyTorch, TensorFlow, and `llama-cpp-python`.

If `llama-cpp-python` fails:

1. Install Microsoft Visual C++ Build Tools.
2. Restart PowerShell.
3. Activate the venv again.
4. Run:

```powershell
pip install llama-cpp-python
pip install -r requirements.txt
```

## 5. Create `.env`

Create a file named `.env` in the project root:

```env
APP_NAME=DiagnoBot
ENVIRONMENT=development
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO

DATABASE_URL=sqlite+aiosqlite:///./diagnobot.db

USE_GPU=false
DEVICE=cpu
MODEL_CACHE_DIR=./models_cache

VISION_MODEL_BACKEND=moondream_gguf
MOONDREAM_VISION_MODEL=salivosa/moondream2-gguf
MOONDREAM_TEXT_FILE=moondream2-q4_k.gguf
MOONDREAM_PROJ_FILE=moondream2-mmproj-f16.gguf

GEMMA_REASONING_MODEL=bartowski/gemma-2-2b-it-GGUF
GEMMA_REASONING_FILE=gemma-2-2b-it-Q4_K_M.gguf

UPLOAD_DIR=./uploads
LOG_FILE=./logs/diagnobot.log
```

If the laptop has an NVIDIA GPU configured with CUDA, change:

```env
USE_GPU=true
DEVICE=cuda
```

Otherwise keep CPU mode.

## 6. First Run

```powershell
python main.py
```

Open:

```text
http://127.0.0.1:8000/dashboard/
http://127.0.0.1:8000/docs
```

The first image analysis may take a long time because Moondream and Gemma GGUF files are downloaded into `models_cache`.

## 7. Faster Setup By Copying Model Cache

If your machine already downloaded models, copy this folder to the teammate laptop:

```text
models_cache
```

Place it inside the cloned project folder:

```text
Diagnobot/models_cache
```

This avoids downloading Moondream/Gemma again.

## 8. Demo Flow

For the conference demo, use this order:

1. Start backend with `python main.py`.
2. Open `/dashboard/`.
3. Upload a clear skin/wound image.
4. Add symptom text like `pain` or `redness`.
5. Run analysis.
6. Show:
   - classifier result
   - visual description
   - triage assessment
   - urgency
   - specialist recommendation
   - reasons and safety note

## 9. Important Notes

- This is a triage decision-support prototype, not a diagnostic tool.
- If the classifier confidence is low, the system should show `Uncertain`.
- If no visible wound/lesion is present, the VLM prompt is designed to say that clearly.
- Moondream/Gemma loading can take time on CPU even with 16GB RAM.
- Close other heavy apps before running the demo.

