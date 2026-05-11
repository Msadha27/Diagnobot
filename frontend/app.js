const state = {
  lastResult: null,
  stream: null,
  capturedBlob: null,
};

const els = {
  apiStatus: document.getElementById("apiStatus"),
  uploadForm: document.getElementById("uploadForm"),
  fileInput: document.getElementById("fileInput"),
  analysisMode: document.getElementById("analysisMode"),
  patientId: document.getElementById("patientId"),
  temperature: document.getElementById("temperature"),
  symptoms: document.getElementById("symptoms"),
  pipelineLabel: document.getElementById("pipelineLabel"),
  summaryBox: document.getElementById("summaryBox"),
  jsonOutput: document.getElementById("jsonOutput"),
  chart: document.getElementById("predictionChart"),
  copyJson: document.getElementById("copyJson"),
  cameraPreview: document.getElementById("cameraPreview"),
  captureCanvas: document.getElementById("captureCanvas"),
  startCamera: document.getElementById("startCamera"),
  captureFrame: document.getElementById("captureFrame"),
  sendCapture: document.getElementById("sendCapture"),
  cameraStatus: document.getElementById("cameraStatus"),
  frameWidth: document.getElementById("frameWidth"),
  frameHeight: document.getElementById("frameHeight"),
  captureState: document.getElementById("captureState"),
  refreshHistory: document.getElementById("refreshHistory"),
  historyList: document.getElementById("historyList"),
};

document.querySelectorAll(".nav-btn").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(`${button.dataset.view}View`).classList.add("active");
  });
});

els.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = els.fileInput.files[0];
  if (!file) return;

  const data = buildFormData(file);
  setBusy("Running analysis...");

  try {
    const response = await fetch("/api/v1/analyze/upload", {
      method: "POST",
      body: data,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Analysis failed");
    renderResult(result);
  } catch (error) {
    renderError(error);
  }
});

els.copyJson.addEventListener("click", async () => {
  await navigator.clipboard.writeText(els.jsonOutput.textContent);
});

els.startCamera.addEventListener("click", async () => {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    els.cameraPreview.srcObject = state.stream;
    els.cameraStatus.textContent = "Camera preview is running.";
    els.cameraPreview.onloadedmetadata = () => {
      els.frameWidth.textContent = els.cameraPreview.videoWidth || 0;
      els.frameHeight.textContent = els.cameraPreview.videoHeight || 0;
    };
  } catch (error) {
    els.cameraStatus.textContent = `Camera error: ${error.message}`;
  }
});

els.captureFrame.addEventListener("click", () => {
  const video = els.cameraPreview;
  if (!video.videoWidth) {
    els.cameraStatus.textContent = "Start the camera before capturing.";
    return;
  }

  const canvas = els.captureCanvas;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    state.capturedBlob = blob;
    els.captureState.textContent = "Yes";
    els.cameraStatus.textContent = "Frame captured. Choose a mode and analyze it.";
  }, "image/jpeg", 0.92);
});

els.sendCapture.addEventListener("click", async () => {
  if (!state.capturedBlob) {
    els.cameraStatus.textContent = "Capture a frame first.";
    return;
  }

  const file = new File([state.capturedBlob], "webcam-capture.jpg", { type: "image/jpeg" });
  const data = buildFormData(file);
  setBusy("Analyzing captured frame...");

  try {
    const response = await fetch("/api/v1/analyze/upload", {
      method: "POST",
      body: data,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Capture analysis failed");
    document.querySelector('[data-view="upload"]').click();
    renderResult(result);
  } catch (error) {
    renderError(error);
  }
});

els.refreshHistory.addEventListener("click", loadHistory);

function buildFormData(file) {
  const data = new FormData();
  data.append("file", file);
  data.append("analysis_mode", els.analysisMode.value);
  if (els.patientId.value.trim()) data.append("patient_id", els.patientId.value.trim());
  if (els.temperature.value) data.append("temperature", els.temperature.value);
  if (els.symptoms.value.trim()) data.append("symptoms", els.symptoms.value.trim());
  return data;
}

function setBusy(message) {
  els.pipelineLabel.textContent = "Running";
  els.summaryBox.textContent = message;
}

function renderResult(result) {
  state.lastResult = result;
  els.pipelineLabel.textContent = result.pipeline || "Complete";
  els.jsonOutput.textContent = JSON.stringify(result, null, 2);
  els.summaryBox.textContent = getSummary(result);
  drawChart(getChartItems(result));
}

function renderError(error) {
  els.pipelineLabel.textContent = "Error";
  els.summaryBox.textContent = error.message;
  els.jsonOutput.textContent = JSON.stringify({ error: error.message }, null, 2);
  drawChart([]);
}

function getSummary(result) {
  const analysis = result.analysis || {};
  const nlp = result.nlp || {};
  const report = result.report || {};
  return (
    analysis.doctor_verdict ||
    analysis.description ||
    nlp.summary ||
    report.summary ||
    "Analysis completed."
  );
}

function getChartItems(result) {
  const analysis = result.analysis || {};
  if (analysis.chart_data?.items) return analysis.chart_data.items;

  const nlp = result.nlp || {};
  if (nlp.symptoms?.length) {
    return nlp.symptoms.slice(0, 6).map((label, index) => ({
      label,
      value: Math.max(95 - index * 10, 45),
    }));
  }

  if (nlp.lab_values) {
    return Object.entries(nlp.lab_values).map(([label, value]) => ({ label, value }));
  }
  return [];
}

function drawChart(items) {
  const canvas = els.chart;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f9fbfc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!items.length) {
    ctx.fillStyle = "#657184";
    ctx.font = "16px Segoe UI";
    ctx.fillText("No chart data yet", 24, 44);
    return;
  }

  const max = Math.max(...items.map((item) => Number(item.value) || 0), 100);
  const barHeight = 26;
  const gap = 16;
  const left = 150;
  const top = 28;
  const width = canvas.width - left - 52;

  ctx.font = "13px Segoe UI";
  items.slice(0, 6).forEach((item, index) => {
    const y = top + index * (barHeight + gap);
    const value = Number(item.value) || 0;
    const barWidth = Math.max(4, (value / max) * width);
    ctx.fillStyle = "#344052";
    ctx.fillText(String(item.label).slice(0, 18), 18, y + 18);
    ctx.fillStyle = "#0f8f83";
    roundRect(ctx, left, y, barWidth, barHeight, 7);
    ctx.fillStyle = "#18202a";
    ctx.fillText(`${Math.round(value)}${max <= 100 ? "%" : ""}`, left + barWidth + 10, y + 18);
  });
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
  ctx.fill();
}

async function checkApi() {
  try {
    const response = await fetch("/");
    const result = await response.json();
    els.apiStatus.textContent = `API ${result.status || "running"}`;
  } catch {
    els.apiStatus.textContent = "API unavailable";
  }
}

async function loadHistory() {
  els.historyList.textContent = "Loading...";
  try {
    const response = await fetch("/api/v1/history?limit=10");
    const result = await response.json();
    const records = result.records || [];
    els.historyList.innerHTML = records.length
      ? records.map(historyItem).join("")
      : '<p class="muted">No records yet.</p>';
  } catch (error) {
    els.historyList.textContent = error.message;
  }
}

function historyItem(record) {
  return `
    <article class="history-item">
      <div>
        <strong>${record.type || "analysis"}</strong>
        <span>${record.input_file || "No file"} · ${record.status}</span>
      </div>
      <span>${record.confidence ?? ""}</span>
    </article>
  `;
}

checkApi();
loadHistory();
drawChart([]);
