document.addEventListener('DOMContentLoaded', () => {
    
    // Live Vitals Elements
    const hrVal = document.getElementById('val-hr');
    const spo2Val = document.getElementById('val-spo2');
    const bpVal = document.getElementById('val-bp');
    const tempVal = document.getElementById('val-temp');
    
    // Base Vitals
    let hr = 78;
    let spo2 = 98;
    let bpSys = 120;
    let bpDia = 80;
    let temp = 98.6;

    let isEmergency = false;

    // Simulate Live Data Fluctuations
    setInterval(() => {
        if(!isEmergency) {
            // Normal fluctuation
            hr = 75 + Math.floor(Math.random() * 8);
            spo2 = 97 + Math.floor(Math.random() * 3);
            bpSys = 118 + Math.floor(Math.random() * 5);
            bpDia = 78 + Math.floor(Math.random() * 5);
            temp = 98.4 + (Math.random() * 0.4);
        } else {
            // Emergency fluctuation
            hr = 130 + Math.floor(Math.random() * 20); // Tachycardia
            spo2 = 88 - Math.floor(Math.random() * 5);  // Hypoxia
            bpSys = 160 + Math.floor(Math.random() * 15);
            bpDia = 100 + Math.floor(Math.random() * 10);
        }

        // Update DOM
        if(hrVal) hrVal.textContent = hr;
        if(spo2Val) spo2Val.textContent = spo2 + '%';
        if(bpVal) bpVal.textContent = `${bpSys}/${bpDia}`;
        if(tempVal) tempVal.textContent = temp.toFixed(1) + '°F';
        
    }, 1500);

    // Trigger Emergency State after 8 seconds for demonstration
    setTimeout(() => {
        isEmergency = true;
        document.body.classList.add('emergency-active');
        
        const alertStatus = document.getElementById('main-status-text');
        if(alertStatus) {
            alertStatus.textContent = "CRITICAL CONDITION DETECTED";
            alertStatus.style.color = "#ff3b30";
        }
        
        // Add critical class to specific boxes
        const hrBox = document.getElementById('box-hr');
        const spo2Box = document.getElementById('box-spo2');
        if(hrBox) hrBox.classList.add('critical');
        if(spo2Box) spo2Box.classList.add('critical');

    }, 8000);

    // AI Prediction Typewriter
    const insightText = "ALERT: Patient showing severe signs of tachycardia and hypoxia. Probability of cardiac event: 89%. Suggested Treatment: Administer Oxygen, prepare defibrillator, notify lead cardiologist immediately. Response time critical: < 2 mins.";
    const typewriterElement = document.getElementById('emergency-ai-insight');
    
    if(typewriterElement) {
        setTimeout(() => {
            typeWriterEffect(typewriterElement, insightText, 0);
        }, 1000);
    }

    function typeWriterEffect(element, text, i) {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            setTimeout(() => typeWriterEffect(element, text, i + 1), 35);
        }
    }

    // Mini Robot Interaction
    const robot = document.getElementById('mini-robot');
    const robotTooltip = document.querySelector('.robot-tooltip');
    
    if(robot && robotTooltip) {
        robot.addEventListener('mouseenter', () => {
            if(isEmergency) {
                robotTooltip.textContent = "Emergency detected! Team notified.";
                robotTooltip.style.borderColor = "#ff3b30";
                robotTooltip.style.color = "#ff3b30";
            } else {
                const phrases = [
                    "Vitals stable.",
                    "AI monitoring active.",
                    "Scanning for anomalies."
                ];
                robotTooltip.textContent = phrases[Math.floor(Math.random() * phrases.length)];
                robotTooltip.style.borderColor = "var(--accent-cyan)";
                robotTooltip.style.color = "#fff";
            }
        });
    }

});
