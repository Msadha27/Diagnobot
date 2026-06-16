document.addEventListener("DOMContentLoaded", () => {
    const stored = localStorage.getItem("diagnobot:lastAnalysis");
    const payload = stored ? JSON.parse(stored) : null;
    const result = payload?.result || {};
    const analysis = result.analysis || {};
    const nlp = result.nlp || {};
    const report = result.report || {};
    const triage = analysis.triage_assessment || {};
    const classification = analysis.classification || {};

    setText("scan-status", payload ? "Analysis Complete" : "No Analysis Loaded");
    setPageCopy();
    renderTags(triage);
    renderInsight(analysis, nlp, report, triage);
    renderMetrics(triage, classification, nlp);
    renderSource(payload);
    renderScanPreview(payload);
    setupJsonViewer(payload);
    animateProgressBars();
});

function setPageCopy() {
    document.title = "AI Triage & Report Center - Ding Dong";
    document.querySelectorAll("span").forEach((span) => {
        if (span.textContent.trim() === "Diagnosis Center") span.textContent = "Triage Center";
    });
    document.querySelectorAll(".section-title").forEach((title) => {
        if (title.textContent.includes("Diagnosis")) title.textContent = "AI Triage & Report Center";
    });
}

function renderTags(triage) {
    const resultTags = document.getElementById("result-tags");
    if (!resultTags) return;
    resultTags.style.display = "block";
    resultTags.innerHTML = `
        <span class="result-tag ${triage.urgency === "emergency" ? "tag-red" : "tag-yellow"}">Urgency: ${(triage.urgency || "review").toUpperCase()}</span>
        <span class="result-tag">Specialist: ${escapeHtml(triage.recommended_specialist || "Clinician review")}</span>
    `;
}

function renderInsight(analysis, nlp, report, triage) {
    const insight = document.getElementById("ai-insight-text");
    if (!insight) return;

    const reasons = [...(triage.red_flags || []), ...(triage.reasons || [])].slice(0, 5);
    const text = [
        analysis.description || nlp.summary || report.summary || "Run an analysis to view generated triage findings.",
        triage.urgency ? `\n\nTriage: ${triage.urgency.toUpperCase()}` : "",
        triage.recommended_specialist ? `\nRecommended specialist: ${triage.recommended_specialist}` : "",
        reasons.length ? `\nReasons: ${reasons.join("; ")}` : "",
        triage.confidence_policy ? `\nPolicy: ${triage.confidence_policy}` : "",
    ].join("");

    insight.textContent = "";
    typeWriterEffect(insight, text, 0);

    const alert = document.getElementById("emergency-alert");
    if (alert) {
        alert.style.display = triage.urgency === "emergency" || triage.urgency === "soon" ? "block" : "none";
        const alertText = alert.querySelector("p");
        if (alertText) {
            alertText.textContent = triage.urgency === "emergency"
                ? "Emergency review is recommended based on detected red flags."
                : "Specialist review is recommended soon based on the combined triage signals.";
        }
    }
}

function renderMetrics(triage, classification, nlp) {
    const riskScore = triage.urgency === "emergency" ? 92 : triage.urgency === "soon" ? 68 : 28;
    const confidence = typeof classification.confidence === "number"
        ? Math.round(classification.confidence * 100)
        : nlp.counts?.abnormal_labs ? Math.min(95, 40 + nlp.counts.abnormal_labs * 10) : 0;

    const progressText = document.querySelectorAll(".progress-text");
    if (progressText[0]) progressText[0].textContent = `${riskScore}%`;
    if (progressText[1]) progressText[1].textContent = confidence ? `${confidence}%` : "--";

    const circles = document.querySelectorAll(".progress-value");
    if (circles[0]) circles[0].setAttribute("data-percent", String(riskScore));
    if (circles[1]) circles[1].setAttribute("data-percent", String(confidence || 1));

    document.querySelectorAll(".metric-label").forEach((label) => {
        if (label.textContent.includes("Heart Rate")) label.textContent = "Classifier Confidence";
    });
}

function renderSource(payload) {
    const sourceBox = [...document.querySelectorAll("p")].find((p) => p.textContent.includes("ECG_Report") || p.textContent.includes("No report"));
    if (sourceBox) sourceBox.textContent = payload?.sourceName || "No source loaded yet";
}

function renderScanPreview(payload) {
    const frame = document.querySelector(".ai-scan-frame");
    if (!frame) return;

    if (payload?.previewImage) {
        frame.style.backgroundImage = `url("${payload.previewImage}")`;
        frame.style.backgroundSize = "contain";
        frame.style.backgroundRepeat = "no-repeat";
        frame.style.backgroundColor = "rgba(0, 0, 0, 0.55)";
    }
}

function setupJsonViewer(payload) {
    const buttons = document.querySelectorAll(".dash-btn");
    const viewButton = [...buttons].find((button) => button.textContent.includes("View Fullscreen"));
    const analyzeButton = [...buttons].find((button) => button.textContent.includes("Analyze Again"));
    const reportCard = viewButton?.closest(".dash-card");
    if (!reportCard) return;

    viewButton.textContent = "View JSON";
    if (analyzeButton) {
        analyzeButton.textContent = "Analyze Again";
        analyzeButton.addEventListener("click", () => { window.location.href = "patient-dashboard.html"; });
    }

    const output = document.createElement("pre");
    output.style.cssText = "display:none;margin-top:1rem;white-space:pre-wrap;max-height:260px;overflow:auto;color:var(--text-muted);background:rgba(0,0,0,0.35);padding:1rem;border-radius:10px;";
    output.textContent = JSON.stringify(payload?.result || { message: "No analysis loaded" }, null, 2);
    reportCard.appendChild(output);

    viewButton.addEventListener("click", () => {
        output.style.display = output.style.display === "none" ? "block" : "none";
    });
}

function animateProgressBars() {
    const circles = document.querySelectorAll(".progress-value");
    circles.forEach((circle) => {
        const percentage = Number(circle.getAttribute("data-percent") || "0");
        const offset = 314 - (314 * percentage / 100);
        circle.style.strokeDashoffset = offset;
        circle.style.stroke = percentage > 85 ? "#ff3b30" : percentage > 50 ? "#ffb400" : "#00f0ff";
    });
}

function typeWriterEffect(element, text, index) {
    if (index < text.length) {
        element.textContent += text.charAt(index);
        setTimeout(() => typeWriterEffect(element, text, index + 1), 14);
    }
}

function setText(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
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
