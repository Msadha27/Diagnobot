document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.mode-card');

    // Add interactive glow effect that follows mouse
    cards.forEach(card => {
        card.addEventListener('mousemove', e => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);
        });
    });

    // Add click animation
    cards.forEach(card => {
        card.addEventListener('click', function(e) {
            // Prevent default just for the demo
            e.preventDefault();
            
            // Add a brief pulse animation class
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
                
                // Demo alert
                const modeName = this.querySelector('.card-title').textContent;
                console.log(`Entering ${modeName}...`);
                
                // Navigate to the respective dashboard
                window.location.href = modeName.includes('Patient') ? 'patient-dashboard.html' : 'doctor-dashboard.html';
            }, 150);
        });
    });
});
