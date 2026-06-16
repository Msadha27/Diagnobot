const API = "";

const state = {
    stream: null,
    capturedBlob: null,
    selectedUploadFile: null,
    selectedUploadMode: "auto",
};

document.addEventListener("DOMContentLoaded", () => {
    setupSidebarTabs();
    setupPatientCamera();
    setupUploadFlow();
    setupHistory();
    setupDoctorSecondOpinion();
    setupDataReset();
});

function setupSidebarTabs() {
    const menuItems = document.querySelectorAll(".menu-item");
    const sections = document.querySelectorAll(".section-container");

    function activateTab(targetId) {
        const target = document.getElementById(targetId);
        if (!target) return;

        menuItems.forEach((entry) => entry.classList.remove("active"));
        sections.forEach((section) => section.classList.remove("active"));
        
        const correspondingMenu = Array.from(menuItems).find(item => item.getAttribute("data-target") === targetId);
        if (correspondingMenu) {
            correspondingMenu.classList.add("active");
        }
        target.classList.add("active");
    }

    menuItems.forEach((item) => {
        item.addEventListener("click", (event) => {
            const targetId = item.getAttribute("data-target");
            if (!targetId) return;

            event.preventDefault();
            activateTab(targetId);
            history.pushState(null, null, `#${targetId.replace("-section", "")}`);
        });
    });

    const handleHash = () => {
        const hash = window.location.hash;
        if (hash) {
            const targetId = `${hash.replace("#", "")}-section`;
            activateTab(targetId);
        }
    };

    handleHash();
    window.addEventListener("hashchange", handleHash);
}

function setupPatientCamera() {
    const scanBtn = document.getElementById("start-scan-btn");
    let analyzeBtn = document.getElementById("analyze-capture-btn");
    const preview = document.querySelector(".camera-preview");
    if (!scanBtn || !preview) return;

    if (!analyzeBtn) {
        analyzeBtn = document.createElement("button");
        analyzeBtn.id = "analyze-capture-btn";
        analyzeBtn.className = "dash-btn";
        analyzeBtn.disabled = true;
        analyzeBtn.style.cssText = "margin-top:0.75rem;opacity:0.55;";
        analyzeBtn.textContent = "Analyze Captured Frame";
        scanBtn.insertAdjacentElement("afterend", analyzeBtn);
    }

    const placeholder = preview.querySelector("svg");
    const video = document.createElement("video");
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    video.style.cssText = "display:none;width:100%;height:100%;object-fit:cover;border-radius:12px;";

    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    canvas.style.display = "none";

    preview.appendChild(video);
    preview.appendChild(canvas);

    const symptomInput = createTextInput("capture-symptoms", "Symptoms e.g. pain, redness, discharge");
    const tempInput = createTextInput("capture-temperature", "Temperature optional", "number");
    scanBtn.parentElement.appendChild(wrapInputs(symptomInput, tempInput));

    scanBtn.addEventListener("click", async () => {
        if (!state.stream) {
            try {
                state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                video.srcObject = state.stream;
                video.style.display = "block";
                if (placeholder) placeholder.style.display = "none";
                scanBtn.textContent = "Camera Running";
                scanBtn.disabled = true;
                scanBtn.style.opacity = "0.7";
                if (analyzeBtn) {
                    analyzeBtn.disabled = false;
                    analyzeBtn.style.opacity = "1";
                }
                showScanMessage("Camera ready. Position the area clearly, then click Analyze Captured Frame.", false, "scan-result");
            } catch (error) {
                showScanMessage(`Camera error: ${error.message}`, true, "scan-result");
            }
            return;
        }
    });

    analyzeBtn?.addEventListener("click", async () => {
        if (!video.videoWidth) {
            showScanMessage("Camera is still starting. Try again in a moment.", true, "scan-result");
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.style.opacity = "0.55";
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(async (blob) => {
            state.capturedBlob = blob;
            const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
            await runUploadAnalysis(file, {
                mode: "skin",
                symptoms: symptomInput.value.trim(),
                temperature: tempInput.value ? Number(tempInput.value) : null,
                sourceName: "webcam-capture.jpg",
                resultCardId: "scan-result",
                resultContentId: "scan-result-content",
            });
            analyzeBtn.disabled = false;
            analyzeBtn.style.opacity = "1";
        }, "image/jpeg", 0.92);
    });
}

function setupUploadFlow() {
    document.querySelectorAll("#upload-btn").forEach((button) => {
        let analyzeButton = document.getElementById("analyze-upload-btn");
        let selectedName = document.getElementById("selected-upload-name");
        const uploadCard = button.closest(".dash-card");
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = ".pdf,.txt,.csv,image/*";
        fileInput.style.display = "none";

        if (!selectedName) {
            selectedName = document.createElement("p");
            selectedName.id = "selected-upload-name";
            selectedName.style.cssText = "color:var(--text-muted);font-size:0.85rem;margin-top:1rem;";
            selectedName.textContent = "No file selected";
            button.insertAdjacentElement("afterend", selectedName);
        }

        if (!analyzeButton) {
            analyzeButton = document.createElement("button");
            analyzeButton.id = "analyze-upload-btn";
            analyzeButton.className = "dash-btn";
            analyzeButton.disabled = true;
            analyzeButton.style.cssText = "width:auto;margin:1rem auto 0;opacity:0.55;";
            analyzeButton.textContent = "Analyze Upload";
            selectedName.insertAdjacentElement("afterend", analyzeButton);
        }

        if (!document.getElementById("upload-result") && uploadCard) {
            const resultCard = document.createElement("div");
            resultCard.id = "upload-result";
            resultCard.className = "dash-card";
            resultCard.style.cssText = "display:none;animation:fadeIn 0.5s;";
            resultCard.innerHTML = `
                <h3 style="color: var(--accent-cyan); margin-bottom: 1rem; font-family: 'Outfit', sans-serif;">Upload Analysis Result</h3>
                <div id="upload-result-content"></div>
            `;
            uploadCard.insertAdjacentElement("afterend", resultCard);
        }

        const modeSelect = document.createElement("select");
        modeSelect.innerHTML = `
            <option value="auto">Auto route</option>
            <option value="skin">Skin / rash</option>
            <option value="wound">Wound</option>
            <option value="eye">Eye color</option>
            <option value="fever">Fever signs</option>
            <option value="xray">Chest X-ray</option>
        `;
        modeSelect.style.cssText = "display:block;margin:0 auto 1rem;padding:0.75rem;border-radius:8px;border:1px solid var(--glass-border);background:rgba(0,0,0,0.35);color:#fff;";

        button.parentElement.insertBefore(modeSelect, button);
        button.parentElement.insertBefore(fileInput, button);

        button.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", () => {
            const file = fileInput.files && fileInput.files[0];
            if (!file) return;
            state.selectedUploadFile = file;
            state.selectedUploadMode = modeSelect.value;
            if (selectedName) selectedName.textContent = file.name;
            if (analyzeButton) {
                analyzeButton.disabled = false;
                analyzeButton.style.opacity = "1";
            }
        });

        modeSelect.addEventListener("change", () => {
            state.selectedUploadMode = modeSelect.value;
        });

        analyzeButton?.addEventListener("click", async () => {
            const file = state.selectedUploadFile;
            if (!file) {
                showScanMessage("Choose a file first.", true, "upload-result", "upload-result-content");
                return;
            }
            analyzeButton.disabled = true;
            analyzeButton.style.opacity = "0.55";
            await runUploadAnalysis(file, {
                mode: state.selectedUploadMode,
                sourceName: file.name,
                resultCardId: "upload-result",
                resultContentId: "upload-result-content",
            });
            analyzeButton.disabled = false;
            analyzeButton.style.opacity = "1";
        });
    });
}

async function runUploadAnalysis(file, options = {}) {
    const resultCardId = options.resultCardId || "scan-result";
    const resultContentId = options.resultContentId || "scan-result-content";
    const resultCard = document.getElementById(resultCardId);
    const resultContent = ensureResultContent(resultCardId, resultContentId);
    const loadingContainer = document.querySelector(".loading-bar-container");
    const loadingBar = document.querySelector(".loading-bar");

    if (resultCard) resultCard.style.display = "block";
    if (resultContent) resultContent.innerHTML = resultMarkup("Running analysis...", "Models may take time on first use.");
    if (loadingContainer) loadingContainer.style.display = "block";
    if (loadingBar) {
        loadingBar.style.width = "15%";
        requestAnimationFrame(() => { loadingBar.style.width = "85%"; });
    }

    const data = new FormData();
    data.append("file", file);
    data.append("analysis_mode", options.mode || "auto");
    if (options.symptoms) data.append("symptoms", options.symptoms);
    if (options.temperature !== null && options.temperature !== undefined) {
        data.append("temperature", String(options.temperature));
    }

    try {
        const response = await fetch(`${API}/api/v1/analyze/upload`, { method: "POST", body: data });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Analysis failed");

        localStorage.setItem("diagnobot:lastAnalysis", JSON.stringify({
            result,
            sourceName: options.sourceName || file.name,
            previewImage: await imagePreviewDataUrl(file),
            createdAt: new Date().toISOString(),
        }));

        if (loadingBar) loadingBar.style.width = "100%";
        renderInlineResult(result, resultCardId, resultContentId);
    } catch (error) {
        if (loadingBar) loadingBar.style.width = "0%";
        if (resultContent) resultContent.innerHTML = resultMarkup("Analysis failed", error.message, true);
    }
}

function imagePreviewDataUrl(file) {
    if (!file || !file.type.startsWith("image/") || file.size > 3 * 1024 * 1024) {
        return Promise.resolve(null);
    }

    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(file);
    });
}

function renderInlineResult(result, resultCardId = "scan-result", resultContentId = "scan-result-content") {
    const resultCard = document.getElementById(resultCardId);
    const resultContent = ensureResultContent(resultCardId, resultContentId);
    if (!resultCard || !resultContent) return;

    resultCard.style.display = "block";
    const analysis = result.analysis || {};
    const nlp = result.nlp || {};
    const report = result.report || {};
    const triage = analysis.triage_assessment || {};
    const classification = analysis.classification || {};
    const isDocument = !!result.nlp || !!result.report;
    const title = isDocument
        ? "Report Analysis Complete"
        : `Triage: ${(triage.urgency || "complete").toUpperCase()}`;
    const details = [
        isDocument ? `Summary: ${escapeHtml(nlp.summary || report.summary || "Clinical text extracted.")}` : "",
        triage.recommended_specialist ? `Specialist: ${escapeHtml(triage.recommended_specialist)}` : "",
        !isDocument ? `Classification: ${escapeHtml(classification.label || classification.disease || "Not classified")}` : "",
        !isDocument ? `Confidence: ${formatPercent(classification.confidence)}` : "",
        analysis.description ? `<strong>Visible findings:</strong> ${escapeHtml(analysis.description)}` : "",
        ...(triage.reasons || []).slice(0, 3),
    ].filter(Boolean).join("<br>");

    resultContent.innerHTML = resultMarkup(title, details);
}

function setupHistory() {
    let historyList = document.getElementById("history-list");
    if (!historyList) {
        const historySection = document.getElementById("history-section");
        const holder = historySection?.querySelector("div[style*='max-width']");
        if (holder) {
            holder.innerHTML = `<div id="history-list"></div>`;
            historyList = document.getElementById("history-list");
        }
    }
    if (!historyList) return;
    loadHistory(historyList);
}

async function loadHistory(container) {
    container.innerHTML = `<div class="history-card"><div><h4>Loading history...</h4><p style="color:var(--text-muted);font-size:0.85rem;">Fetching saved analyses</p></div></div>`;
    try {
        const response = await fetch(`${API}/api/v1/history?limit=8`);
        const data = await response.json();
        const records = data.records || [];
        if (!records.length) {
            container.innerHTML = `<div class="history-card"><div><h4>No records yet</h4><p style="color:var(--text-muted);font-size:0.85rem;">Run an image or report analysis to populate this area.</p></div><span class="status-badge status-pending">Waiting</span></div>`;
            return;
        }
        container.innerHTML = records.map((record) => `
            <div class="history-card">
                <div>
                    <h4 style="margin-bottom:0.25rem;">${escapeHtml(record.type || "analysis")}</h4>
                    <p style="color:var(--text-muted);font-size:0.85rem;">${escapeHtml(record.input_file || "No file")} - ${escapeHtml(record.status || "saved")}</p>
                </div>
                <span class="status-badge ${record.status === "success" ? "status-success" : "status-warning"}">${escapeHtml(record.status || "saved")}</span>
            </div>
        `).join("");
    } catch (error) {
        container.innerHTML = `<div class="history-card"><div><h4>History unavailable</h4><p style="color:var(--text-muted);font-size:0.85rem;">${escapeHtml(error.message)}</p></div><span class="status-badge status-warning">Offline</span></div>`;
    }
}

function setupDoctorSecondOpinion() {
    const analyzeSection = document.getElementById("analyze-section");
    if (!analyzeSection || !document.body.textContent.includes("Clinical Analysis Dashboard")) return;

    const card = document.createElement("div");
    card.className = "dash-card";
    card.style.marginTop = "2rem";
    card.innerHTML = `
        <h3 style="margin-bottom:1rem;color:var(--accent-blue);font-family:'Outfit',sans-serif;">Doctor Second Opinion</h3>
        <textarea id="doctor-findings" rows="3" placeholder="Clinical findings, image summary, abnormal labs..." style="width:100%;padding:1rem;border-radius:10px;border:1px solid var(--glass-border);background:rgba(0,0,0,0.35);color:#fff;margin-bottom:1rem;"></textarea>
        <textarea id="doctor-plan" rows="3" placeholder="Doctor decision or treatment plan..." style="width:100%;padding:1rem;border-radius:10px;border:1px solid var(--glass-border);background:rgba(0,0,0,0.35);color:#fff;margin-bottom:1rem;"></textarea>
        <button id="doctor-opinion-btn" class="dash-btn doctor-theme">Generate Second Opinion</button>
        <div id="doctor-opinion-result" style="margin-top:1rem;color:var(--text-muted);line-height:1.6;"></div>
    `;
    analyzeSection.appendChild(card);

    document.getElementById("doctor-opinion-btn").addEventListener("click", async () => {
        const clinical_findings = document.getElementById("doctor-findings").value.trim();
        const doctor_plan = document.getElementById("doctor-plan").value.trim();
        const output = document.getElementById("doctor-opinion-result");
        if (!clinical_findings && !doctor_plan) {
            output.textContent = "Enter findings or a plan first.";
            return;
        }
        output.textContent = "Reviewing...";
        try {
            const response = await fetch(`${API}/api/v1/doctor/second-opinion`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ clinical_findings, doctor_plan }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Second opinion failed");
            const opinion = data.second_opinion || {};
            output.innerHTML = `
                <strong style="color:#fff;">${escapeHtml(opinion.summary || "Second opinion generated.")}</strong>
                <ul style="margin-top:0.8rem;">${(opinion.cautions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
            `;
        } catch (error) {
            output.textContent = error.message;
        }
    });
}

function setupDataReset() {
    document.querySelectorAll(".js-reset-db").forEach((button) => {
        button.addEventListener("click", async (event) => {
            event.preventDefault();
            button.style.pointerEvents = "none";
            const oldText = button.querySelector("span")?.textContent || "New Analysis DB";
            const label = button.querySelector("span");
            if (label) label.textContent = "Resetting...";

            try {
                const response = await fetch(`${API}/api/v1/admin/reset-data`, { method: "POST" });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || "Reset failed");

                localStorage.removeItem("diagnobot:lastAnalysis");
                state.capturedBlob = null;
                state.selectedUploadFile = null;

                const historyList = document.getElementById("history-list");
                if (historyList) await loadHistory(historyList);

                showDashboardNotice(
                    "Fresh database ready",
                    "Old analysis records, reports, uploaded-file rows, and browser cached result were cleared."
                );
            } catch (error) {
                showDashboardNotice("Reset failed", error.message, true);
            } finally {
                if (label) label.textContent = oldText;
                button.style.pointerEvents = "";
            }
        });
    });
}

function showDashboardNotice(title, body, isError = false) {
    let notice = document.getElementById("dashboard-notice");
    if (!notice) {
        notice = document.createElement("div");
        notice.id = "dashboard-notice";
        notice.style.cssText = "position:fixed;right:1.25rem;bottom:1.25rem;z-index:50;max-width:360px;padding:1rem 1.1rem;border-radius:14px;border:1px solid var(--glass-border);background:rgba(5,10,20,0.92);box-shadow:0 18px 45px rgba(0,0,0,0.35);backdrop-filter:blur(16px);";
        document.body.appendChild(notice);
    }

    notice.innerHTML = `
        <strong style="display:block;color:${isError ? "#ff3b30" : "#00ff80"};margin-bottom:0.35rem;">${escapeHtml(title)}</strong>
        <span style="display:block;color:var(--text-muted);font-size:0.9rem;line-height:1.45;">${escapeHtml(body)}</span>
    `;
    clearTimeout(showDashboardNotice.timer);
    showDashboardNotice.timer = setTimeout(() => notice.remove(), 4200);
}

function showScanMessage(title, isError = false, resultCardId = "scan-result", resultContentId = "scan-result-content") {
    const resultCard = document.getElementById(resultCardId);
    const resultContent = ensureResultContent(resultCardId, resultContentId);
    if (!resultCard || !resultContent) return;
    resultCard.style.display = "block";
    resultContent.innerHTML = resultMarkup(title, "", isError);
}

function ensureResultContent(resultCardId = "scan-result", resultContentId = "scan-result-content") {
    const resultCard = document.getElementById(resultCardId);
    if (!resultCard) return null;
    let resultContent = document.getElementById(resultContentId);
    if (!resultContent) {
        resultCard.innerHTML = `
            <h3 style="color: var(--accent-cyan); margin-bottom: 1rem; font-family: 'Outfit', sans-serif;">Analysis Result</h3>
            <div id="${resultContentId}"></div>
        `;
        resultContent = document.getElementById(resultContentId);
    }
    return resultContent;
}

function resultMarkup(title, body, isError = false) {
    return `
        <p style="font-size:1.2rem;font-weight:700;color:${isError ? "#ff3b30" : "#00ff80"};margin-bottom:0.75rem;">${escapeHtml(title)}</p>
        <p style="color:var(--text-muted);font-size:0.95rem;line-height:1.6;padding:1rem;background:rgba(0,240,255,0.05);border-left:2px solid var(--accent-cyan);">${body}</p>
    `;
}

function createTextInput(id, placeholder, type = "text") {
    const input = document.createElement("input");
    input.id = id;
    input.type = type;
    input.placeholder = placeholder;
    input.style.cssText = "padding:0.8rem;border-radius:8px;border:1px solid var(--glass-border);background:rgba(0,0,0,0.35);color:#fff;min-width:0;";
    return input;
}

function wrapInputs(...inputs) {
    const wrapper = document.createElement("div");
    wrapper.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-top:0.75rem;";
    inputs.forEach((input) => wrapper.appendChild(input));
    return wrapper;
}

function formatPercent(value) {
    return typeof value === "number" ? `${Math.round(value * 100)}%` : "n/a";
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
    }[char]));
}
