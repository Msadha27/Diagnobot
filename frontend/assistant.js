document.addEventListener("DOMContentLoaded", () => {
    const chatContainer = document.getElementById("chat-window");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");

    async function handleSend() {
        const text = chatInput.value.trim();
        if (!text) return;

        addMessage(text, "user");
        chatInput.value = "";
        showTypingIndicator();

        try {
            const modelResponse = await fetch("/api/v1/assistant/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text }),
            });
            const modelData = await modelResponse.json();
            if (!modelResponse.ok) throw new Error(modelData.detail || "Local model request failed");

            let triageData = null;
            try {
                const triageResponse = await fetch("/api/v1/triage/emergency", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ symptoms: text }),
                });
                const result = await triageResponse.json();
                if (triageResponse.ok) triageData = result;
            } catch (_) {
                // Keep the actual model answer if the separate rule service is unavailable.
            }

            addStructuredResponse(modelData.answer, triageData, modelData.disclaimer);
        } catch (error) {
            addMessage(`No model answer was generated. ${error.message}`, "ai");
        }
    }

    function addStructuredResponse(answer, triageData, disclaimer) {
        const indicator = document.getElementById("typing-indicator");
        if (indicator) indicator.remove();

        const bubble = document.createElement("div");
        bubble.classList.add("chat-bubble", "chat-ai", "chat-ai-structured");

        const triage = triageData?.triage_assessment || {};
        const urgency = String(triageData?.urgency || "unavailable").toLowerCase();
        const alertLevel = ["soon", "emergency", "severe"].includes(urgency);
        const flag = document.createElement("div");
        flag.className = `chat-triage-flag ${alertLevel ? "chat-triage-flag--alert" : ""}`;
        flag.textContent = `TRIAGE: ${urgency.toUpperCase()}`;
        bubble.appendChild(flag);

        const reasons = [...new Set([
            ...(triage.red_flags || []),
            ...(triage.reasons || []),
            triageData?.reason || "",
        ].map((item) => String(item).trim()).filter(Boolean))].slice(0, 4);
        const reasonHeading = document.createElement("strong");
        reasonHeading.className = "chat-response-heading";
        reasonHeading.textContent = "Reason";
        bubble.appendChild(reasonHeading);

        const reasonList = document.createElement("ul");
        reasonList.className = "chat-reason-list";
        (reasons.length ? reasons : [triageData ? "No rule-based warning signal was detected in the entered text." : "Rule-based triage was unavailable."]).forEach((reason) => {
            const item = document.createElement("li");
            item.textContent = reason;
            reasonList.appendChild(item);
        });
        bubble.appendChild(reasonList);

        const responseHeading = document.createElement("strong");
        responseHeading.className = "chat-response-heading";
        responseHeading.textContent = "Response";
        bubble.appendChild(responseHeading);

        const answerText = document.createElement("p");
        answerText.className = "chat-model-answer";
        answerText.textContent = cleanModelText(answer);
        bubble.appendChild(answerText);

        if (disclaimer) {
            const note = document.createElement("p");
            note.className = "chat-prototype-note";
            note.textContent = disclaimer;
            bubble.appendChild(note);
        }

        chatContainer.appendChild(bubble);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function cleanModelText(value) {
        return String(value || "")
            .replace(/^\s*\*\s+/gm, "• ")
            .replace(/\*/g, "")
            .replace(/__/g, "")
            .replace(/^\s{0,3}#{1,6}\s*/gm, "")
            .trim();
    }

    function addMessage(text, sender) {
        const indicator = document.getElementById("typing-indicator");
        if (indicator) indicator.remove();
        const bubble = document.createElement("div");
        bubble.classList.add("chat-bubble", sender === "user" ? "chat-user" : "chat-ai");
        bubble.textContent = text;
        chatContainer.appendChild(bubble);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function showTypingIndicator() {
        const bubble = document.createElement("div");
        bubble.classList.add("chat-bubble", "chat-ai");
        bubble.id = "typing-indicator";
        bubble.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
        chatContainer.appendChild(bubble);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    if (sendBtn) sendBtn.addEventListener("click", handleSend);
    if (chatInput) {
        chatInput.addEventListener("keypress", (event) => {
            if (event.key === "Enter") handleSend();
        });
    }

    setupVoiceAssistant(addMessage);
    setupSymptomAnalyzer();
    setupFaqs();
});

function setupVoiceAssistant(addMessage) {
    const voiceMic = document.getElementById("voice-mic");
    const equalizer = document.getElementById("voice-eq");
    const voiceStatus = document.getElementById("voice-status");
    if (!voiceMic) return;

    voiceMic.addEventListener("click", () => {
        if (voiceStatus) {
            voiceStatus.textContent = "Voice input is not connected";
            voiceStatus.style.color = "var(--text-muted)";
        }
        addMessage("Voice input is not connected yet. Type a message to get a local Llama 3.2 response.", "ai");
    });
}

function setupSymptomAnalyzer() {
    const bodyParts = document.querySelectorAll(".body-part");
    const symptomTitle = document.getElementById("symptom-title");
    const symptomDesc = document.getElementById("symptom-desc");
    const riskBadge = document.getElementById("risk-badge");
    const symptomData = {
        head: { title: "Head / Cranial", desc: "Possible triage context: headache, eye strain, fever, dehydration.", risk: "Review if severe", color: "#ffb400" },
        chest: { title: "Chest / Breathing", desc: "Chest pain or breathing difficulty should be treated as an emergency red flag.", risk: "High Risk", color: "#ff3b30" },
        stomach: { title: "Abdomen", desc: "Track pain, vomiting, fever, dehydration, and report abnormalities.", risk: "Monitor", color: "#00ff80" },
    };
    bodyParts.forEach((part) => {
        part.addEventListener("click", () => {
            bodyParts.forEach((node) => node.classList.remove("active"));
            part.classList.add("active");
            const data = symptomData[part.getAttribute("data-area")];
            if (!data) return;
            symptomTitle.textContent = data.title;
            symptomDesc.textContent = data.desc;
            riskBadge.textContent = data.risk;
            riskBadge.style.color = data.color;
            riskBadge.style.borderColor = data.color;
            riskBadge.style.background = `${data.color}22`;
        });
    });
}

function setupFaqs() {
    document.querySelectorAll(".faq-card").forEach((faq) => {
        faq.querySelector(".faq-header")?.addEventListener("click", () => faq.classList.toggle("open"));
    });
}
