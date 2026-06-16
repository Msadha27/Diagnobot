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
            const response = await fetch("/api/v1/triage/emergency", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symptoms: text }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Triage failed");
            const triage = data.triage_assessment || {};
            const reply = [
                data.patient_summary || "I reviewed your symptoms.",
                triage.recommended_specialist ? `Recommended specialist: ${triage.recommended_specialist}.` : "",
                triage.next_steps?.length ? `Next step: ${triage.next_steps[0]}` : "",
                "This is triage support only, not a diagnosis.",
            ].filter(Boolean).join(" ");
            addMessage(reply, "ai");
        } catch (error) {
            addMessage(`I could not complete triage right now: ${error.message}`, "ai");
        }
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
        const isActive = voiceMic.classList.toggle("active");
        equalizer?.classList.toggle("active", isActive);
        if (voiceStatus) {
            voiceStatus.textContent = isActive ? "Listening..." : "Voice Assistant Ready";
            voiceStatus.style.color = isActive ? "var(--accent-cyan)" : "var(--text-muted)";
        }
        if (isActive) {
            setTimeout(() => {
                voiceMic.classList.remove("active");
                equalizer?.classList.remove("active");
                if (voiceStatus) voiceStatus.textContent = "Voice Assistant Ready";
                addMessage("Voice mode is ready for demo. Please type symptoms for backend triage.", "ai");
            }, 2500);
        }
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
