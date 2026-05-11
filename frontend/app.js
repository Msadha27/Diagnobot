const state = {
  lastResult: null,
  stream: null,
  capturedBlob: null,
  activeRequest: null,
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
  stopCamera: document.getElementById("stopCamera"),
  captureFrame: document.getElementById("captureFrame"),
  sendCapture: document.getElementById("sendCapture"),
  cameraStatus: document.getElementById("cameraStatus"),
  frameWidth: document.getElementById("frameWidth"),
  frameHeight: document.getElementById("frameHeight"),
  captureState: document.getElementById("captureState"),
  refreshHistory: document.getElementById("refreshHistory"),
  historyList: document.getElementById("historyList"),
  triageForm: document.getElementById("triageForm"),
  triageSymptoms: document.getElementById("triageSymptoms"),
  triageTemperature: document.getElementById("triageTemperature"),
  triageVisual: document.getElementById("triageVisual"),
  triageUrgency: document.getElementById("triageUrgency"),
  triageSummary: document.getElementById("triageSummary"),
  specialistCard: document.getElementById("specialistCard"),
  useLastResult: document.getElementById("useLastResult"),
  doctorForm: document.getElementById("doctorForm"),
  doctorPatientId: document.getElementById("doctorPatientId"),
  doctorFindings: document.getElementById("doctorFindings"),
  doctorPlan: document.getElementById("doctorPlan"),
  doctorHistory: document.getElementById("doctorHistory"),
  doctorStatus: document.getElementById("doctorStatus"),
  doctorSummary: document.getElementById("doctorSummary"),
  doctorChecks: document.getElementById("doctorChecks"),
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
  setBusy("Sending file to backend /api/v1/analyze/upload...");

  try {
    const response = await fetchWithTimeout("/api/v1/analyze/upload", data);
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

els.stopCamera.addEventListener("click", stopCamera);

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
  setBusy("Sending captured frame to backend /api/v1/analyze/upload...");

  try {
    const response = await fetchWithTimeout("/api/v1/analyze/upload", data);
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Capture analysis failed");
    document.querySelector('[data-view="upload"]').click();
    renderResult(result);
  } catch (error) {
    renderError(error);
  }
});

els.refreshHistory.addEventListener("click", loadHistory);

els.useLastResult.addEventListener("click", () => {
  const text = latestResultText();
  els.triageVisual.value = text.slice(0, 260);
  els.triageSummary.textContent = text || "No previous result available yet.";
});

els.triageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  els.triageUrgency.textContent = "Running";
  els.triageSummary.textContent = "Generating triage and specialist recommendation...";

  try {
    const response = await postJson("/api/v1/triage/emergency", {
      symptoms: els.triageSymptoms.value.trim(),
      visual_summary: els.triageVisual.value.trim(),
      temperature: els.triageTemperature.value ? Number(els.triageTemperature.value) : null,
    });
    renderTriage(response);
  } catch (error) {
    els.triageUrgency.textContent = "Error";
    els.triageSummary.textContent = error.message;
  }
});

els.doctorForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  els.doctorStatus.textContent = "Running";
  els.doctorSummary.textContent = "Reviewing entered clinical decision...";

  try {
    const response = await postJson("/api/v1/doctor/second-opinion", {
      patient_id: els.doctorPatientId.value.trim() || null,
      clinical_findings: els.doctorFindings.value.trim(),
      doctor_plan: els.doctorPlan.value.trim(),
      patient_history: els.doctorHistory.value.trim() || null,
    });
    renderDoctorOpinion(response);
  } catch (error) {
    els.doctorStatus.textContent = "Error";
    els.doctorSummary.textContent = error.message;
  }
});

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
  els.summaryBox.textContent = `${message}\nIf this is the first image analysis, model loading on CPU can take a while. Watch the terminal for backend logs.`;
  setButtonsDisabled(true);
}

function renderResult(result) {
  state.lastResult = result;
  els.pipelineLabel.textContent = result.pipeline || "Complete";
  els.jsonOutput.textContent = JSON.stringify(result, null, 2);
  els.summaryBox.textContent = getSummary(result);
  drawChart(getChartItems(result));
  setButtonsDisabled(false);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Request failed");
  return result;
}

function renderTriage(result) {
  els.triageUrgency.textContent = result.urgency;
  els.triageSummary.textContent = result.patient_summary;
  els.specialistCard.innerHTML = `
    <strong>${result.recommended_specialist}</strong>
    <span>${result.reason}</span>
    <ul>${result.next_steps.map((item) => `<li>${item}</li>`).join("")}</ul>
  `;
  els.jsonOutput.textContent = JSON.stringify(result, null, 2);
}

function renderDoctorOpinion(result) {
  const opinion = result.second_opinion || {};
  els.doctorStatus.textContent = "Complete";
  els.doctorSummary.textContent = opinion.summary || "Second opinion generated.";
  els.doctorChecks.innerHTML = `
    <h3>Cautions</h3>
    <ul>${(opinion.cautions || []).map((item) => `<li>${item}</li>`).join("")}</ul>
    <h3>Suggested checks</h3>
    <ul>${(opinion.suggested_checks || []).map((item) => `<li>${item}</li>`).join("")}</ul>
  `;
  els.jsonOutput.textContent = JSON.stringify(result, null, 2);
}

function renderError(error) {
  els.pipelineLabel.textContent = "Error";
  els.summaryBox.textContent = error.message;
  els.jsonOutput.textContent = JSON.stringify({ error: error.message }, null, 2);
  drawChart([]);
  setButtonsDisabled(false);
}

async function fetchWithTimeout(url, body, timeoutMs = 900000) {
  if (state.activeRequest) {
    state.activeRequest.abort();
  }
  const controller = new AbortController();
  state.activeRequest = controller;
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      method: "POST",
      body,
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Backend request timed out after 15 minutes. Check the terminal logs to see whether the model is still running.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
    state.activeRequest = null;
  }
}

function setButtonsDisabled(disabled) {
  els.uploadForm.querySelector("button[type='submit']").disabled = disabled;
  els.sendCapture.disabled = disabled;
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  els.cameraPreview.srcObject = null;
  state.capturedBlob = null;
  els.frameWidth.textContent = "0";
  els.frameHeight.textContent = "0";
  els.captureState.textContent = "No";
  els.cameraStatus.textContent = "Camera stopped.";
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

function latestResultText() {
  const result = state.lastResult || {};
  const analysis = result.analysis || {};
  const nlp = result.nlp || {};
  const report = result.report || {};
  if (analysis.doctor_verdict) return analysis.doctor_verdict;
  if (analysis.description) return analysis.description;
  if (nlp.summary) return nlp.summary;
  if (report.summary) return report.summary;
  return "";
}

function getChartItems(result) {
  const analysis = result.analysis || {};
  if (analysis.chart_data?.items) return analysis.chart_data.items;

  const nlp = result.nlp || {};
  if (nlp.important_findings?.length) {
    return nlp.important_findings.slice(0, 8).map((lab) => ({
      label: lab.name,
      value: abnormalityScore(lab),
      displayValue: `${lab.value} ${lab.unit || ""} (${lab.status})`,
    }));
  }

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

function abnormalityScore(lab) {
  if (lab.severity === "high") return 92;
  if (lab.severity === "medium") return 72;
  if (lab.severity === "low") return 45;
  return 25;
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
    ctx.fillText(item.displayValue || `${Math.round(value)}${max <= 100 ? "%" : ""}`, left + barWidth + 10, y + 18);
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
    els.apiStatus.textContent = `API ${result.status || "running"} on ${window.location.origin}`;
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
