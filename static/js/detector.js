// Exoplanet Detection System - Main Logic
class ExoplanetDetector {
    constructor() {
        this.initializeEventListeners();
        this.startMissionClock();
        this.updateSliderValues();
    }

    initializeEventListeners() {
        // Slider value updates
        const sliders = document.querySelectorAll('.cosmic-slider');
        sliders.forEach(slider => {
            slider.addEventListener('input', (e) => {
                this.updateSliderValue(e.target);
                this.updateParameterPreview();
            });
        });

        // Detection button
        const detectBtn = document.getElementById('detect-btn');
        detectBtn.addEventListener('click', () => this.initiateDetection());

        // Toggle switches
        const toggles = document.querySelectorAll('.sci-fi-toggle');
        toggles.forEach(toggle => {
            toggle.addEventListener('change', () => this.updateParameterPreview());
        });
    }

    updateSliderValue(slider) {
        const valueElement = document.getElementById(slider.id + '-value');
        if (valueElement) {
            valueElement.textContent = parseFloat(slider.value).toFixed(2);
        }
    }

    updateSliderValues() {
        const sliders = document.querySelectorAll('.cosmic-slider');
        sliders.forEach(slider => this.updateSliderValue(slider));
    }

    updateParameterPreview() {
        // Could add real-time parameter visualization here
        console.log('Parameters updated:', this.getCurrentParameters());
    }

    getCurrentParameters() {
        return {
            period: parseFloat(document.getElementById('period').value),
            depth: parseFloat(document.getElementById('depth').value),
            duration: parseFloat(document.getElementById('duration').value),
            impact: parseFloat(document.getElementById('impact').value),
            teq: parseFloat(document.getElementById('teq').value),
            snr: parseFloat(document.getElementById('snr').value),
            steff: parseFloat(document.getElementById('steff').value),
            slogg: parseFloat(document.getElementById('slogg').value),
            fp_nt: document.getElementById('fp_nt').checked ? 1 : 0,
            fp_ss: document.getElementById('fp_ss').checked ? 1 : 0,
            fp_co: document.getElementById('fp_co').checked ? 1 : 0,
            fp_ec: document.getElementById('fp_ec').checked ? 1 : 0
        };
    }

    async initiateDetection() {
        const parameters = this.getCurrentParameters();
        const detectBtn = document.getElementById('detect-btn');
        const waitingState = document.querySelector('.waiting-state');
        const resultsContent = document.getElementById('results-content');

        // Show scanning state
        detectBtn.disabled = true;
        detectBtn.innerHTML = '<i class="fas fa-sync fa-spin"></i><span>ANALYZING CELESTIAL DATA...</span>';
        waitingState.style.display = 'none';
        resultsContent.style.display = 'block';
        resultsContent.innerHTML = this.createScanningTemplate();

        try {
            // Send request to Flask backend
            const response = await fetch('/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(parameters)
            });

            const result = await response.json();

            if (result.success) {
                this.displayResults(result);
            } else {
                this.displayError(result.error);
            }

        } catch (error) {
            this.displayError('Network error: ' + error.message);
        } finally {
            // Reset button
            detectBtn.disabled = false;
            detectBtn.innerHTML = '<div class="button-glow"></div><i class="fas fa-search"></i><span>INITIATE PLANET ANALYSIS</span><div class="button-pulse"></div>';
        }
    }

    createScanningTemplate() {
        return `
            <div class="scanning-results">
                <div class="scan-animation-large">
                    <div class="quantum-scanner"></div>
                    <div class="data-stream">
                        <div class="stream-line"></div>
                        <div class="stream-line delay-1"></div>
                        <div class="stream-line delay-2"></div>
                    </div>
                </div>
                <h3>PROCESSING CELESTIAL DATA</h3>
                <p>AI model analyzing planetary signatures...</p>
                <div class="progress-indicator">
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                </div>
            </div>
        `;
    }

    displayResults(result) {
        const resultsContent = document.getElementById('results-content');
        const isPlanet = result.prediction === 1;
        const confidence = result.confidence;
        
        resultsContent.innerHTML = this.createResultsTemplate(result, isPlanet, confidence);
        this.animateResults();
    }

    createResultsTemplate(result, isPlanet, confidence) {
        const confidenceClass = confidence > 0.7 ? 'confidence-high' : 
                              confidence > 0.4 ? 'confidence-medium' : 'confidence-low';
        
        const confidenceColor = confidence > 0.7 ? '#4db6ac' : 
                               confidence > 0.4 ? '#ff9800' : '#f44336';

        return `
            <div class="result-card ${isPlanet ? 'result-planet' : 'result-fake'}">
                <div class="result-header">
                    <i class="fas ${isPlanet ? 'fa-globe-americas' : 'fa-times-circle'} result-icon"></i>
                    <h2 class="result-title">
                        ${isPlanet ? 'PLANET DETECTED' : 'NO PLANET SIGNATURE'}
                    </h2>
                </div>
                
                <div class="result-message">
                    <p>${this.getResultMessage(isPlanet, confidence)}</p>
                </div>

                <div class="confidence-display">
                    <div class="confidence-header">
                        <span>AI CONFIDENCE LEVEL</span>
                        <span class="confidence-value">${Math.round(confidence * 100)}%</span>
                    </div>
                    <div class="confidence-meter">
                        <div class="confidence-fill ${confidenceClass}" 
                             style="width: ${confidence * 100}%"></div>
                    </div>
                </div>

                <div class="key-factors">
                    <h4>KEY ANALYSIS FACTORS:</h4>
                    <ul>
                        ${result.key_factors.map(factor => `<li>${factor}</li>`).join('')}
                    </ul>
                </div>

                <div class="feature-importance">
                    <h4>FEATURE CONTRIBUTION:</h4>
                    <div class="importance-bars">
                        ${this.createImportanceBars(result.feature_importance)}
                    </div>
                </div>
            </div>
        `;
    }

    getResultMessage(isPlanet, confidence) {
        if (isPlanet) {
            if (confidence > 0.8) return "Strong planetary signatures detected! This candidate exhibits excellent characteristics for a confirmed exoplanet.";
            if (confidence > 0.6) return "Promising planetary candidate detected. Shows good transit characteristics and orbital stability.";
            return "Potential planetary signature detected. Further observation recommended for confirmation.";
        } else {
            if (confidence < 0.2) return "Clear false positive signature. Shows characteristics inconsistent with planetary transits.";
            if (confidence < 0.4) return "Unlikely to be planetary. Signal patterns suggest stellar variability or instrumental artifact.";
            return "Ambiguous signal detected. Some planetary-like features but significant inconsistencies present.";
        }
    }

    createImportanceBars(importance) {
        const maxImportance = Math.max(...Object.values(importance));
        let barsHTML = '';

        for (const [feature, value] of Object.entries(importance)) {
            const percentage = (value / maxImportance) * 100;
            const displayName = this.formatFeatureName(feature);
            
            barsHTML += `
                <div class="importance-item">
                    <span class="importance-label">${displayName}</span>
                    <div class="importance-bar">
                        <div class="importance-fill" style="width: ${percentage}%"></div>
                    </div>
                    <span class="importance-value">${(value * 100).toFixed(1)}%</span>
                </div>
            `;
        }

        return barsHTML;
    }

    formatFeatureName(feature) {
        const names = {
            'koi_period': 'ORBITAL PERIOD',
            'koi_depth': 'TRANSIT DEPTH', 
            'koi_duration': 'TRANSIT DURATION',
            'koi_impact': 'IMPACT PARAMETER',
            'koi_teq': 'EQUILIBRIUM TEMP',
            'koi_model_snr': 'SIGNAL-TO-NOISE',
            'koi_steff': 'STAR TEMPERATURE',
            'koi_slogg': 'SURFACE GRAVITY',
            'koi_fpflag_nt': 'NOT TRANSIT-LIKE',
            'koi_fpflag_ss': 'STELLAR ECLIPSE',
            'koi_fpflag_co': 'CENTROID OFFSET',
            'koi_fpflag_ec': 'EPHEMERIS MATCH'
        };
        
        return names[feature] || feature.replace('koi_', '').toUpperCase();
    }

    animateResults() {
        // Animate confidence meter
        setTimeout(() => {
            const confidenceFill = document.querySelector('.confidence-fill');
            confidenceFill.style.transition = 'width 2s ease-out';
        }, 100);

        // Animate importance bars
        setTimeout(() => {
            const importanceFills = document.querySelectorAll('.importance-fill');
            importanceFills.forEach((fill, index) => {
                setTimeout(() => {
                    fill.style.transition = 'width 1s ease-out';
                }, index * 100);
            });
        }, 500);
    }

    displayError(error) {
        const resultsContent = document.getElementById('results-content');
        resultsContent.innerHTML = `
            <div class="result-card result-error">
                <div class="result-header">
                    <i class="fas fa-exclamation-triangle result-icon"></i>
                    <h2 class="result-title">ANALYSIS ERROR</h2>
                </div>
                <div class="error-message">
                    <p>Unable to process detection request:</p>
                    <p class="error-detail">${error}</p>
                </div>
                <button class="quantum-button" onclick="detector.initiateDetection()">
                    <i class="fas fa-redo"></i>
                    <span>RETRY ANALYSIS</span>
                </button>
            </div>
        `;
    }

    startMissionClock() {
        const clockElement = document.getElementById('mission-clock');
        let startTime = Date.now();

        setInterval(() => {
            const elapsed = Date.now() - startTime;
            const hours = Math.floor(elapsed / 3600000).toString().padStart(2, '0');
            const minutes = Math.floor((elapsed % 3600000) / 60000).toString().padStart(2, '0');
            const seconds = Math.floor((elapsed % 60000) / 1000).toString().padStart(2, '0');
            
            clockElement.textContent = `${hours}:${minutes}:${seconds}`;
        }, 1000);
    }
}

// Additional CSS for dynamic elements
const additionalStyles = `
    .scanning-results {
        text-align: center;
        padding: 40px 20px;
    }

    .scan-animation-large {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto 30px;
    }

    .quantum-scanner {
        width: 100%;
        height: 100%;
        border: 3px solid rgba(79, 195, 247, 0.3);
        border-radius: 50%;
        position: relative;
        overflow: hidden;
    }

    .quantum-scanner::before {
        content: '';
        position: absolute;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--cosmic-blue), transparent);
        animation: quantum-scan 2s linear infinite;
    }

    .data-stream {
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 2px;
        background: rgba(79, 195, 247, 0.1);
        overflow: hidden;
    }

    .stream-line {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, var(--cosmic-blue), transparent);
        animation: data-flow 1.5s linear infinite;
    }

    .delay-1 { animation-delay: 0.5s; }
    .delay-2 { animation-delay: 1s; }

    .progress-indicator {
        margin-top: 20px;
    }

    .progress-bar {
        width: 100%;
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
        overflow: hidden;
    }

    .progress-fill {
        width: 100%;
        height: 100%;
        background: var(--cosmic-blue);
        animation: progress-scan 2s ease-in-out infinite;
    }

    .confidence-display {
        margin: 20px 0;
    }

    .confidence-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }

    .confidence-value {
        font-weight: bold;
        color: var(--cosmic-blue);
    }

    .key-factors, .feature-importance {
        margin: 25px 0;
    }

    .key-factors h4, .feature-importance h4 {
        color: var(--cosmic-blue);
        margin-bottom: 15px;
        font-size: 0.9rem;
    }

    .key-factors ul {
        list-style: none;
        padding: 0;
    }

    .key-factors li {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.85rem;
    }

    .key-factors li:before {
        content: '▸';
        color: var(--cosmic-blue);
        margin-right: 10px;
    }

    .importance-item {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 12px;
        font-size: 0.8rem;
    }

    .importance-label {
        min-width: 150px;
        color: #e3f2fd;
    }

    .importance-bar {
        flex: 1;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        overflow: hidden;
    }

    .importance-fill {
        height: 100%;
        background: var(--cosmic-blue);
        width: 0;
        transition: none;
    }

    .importance-value {
        min-width: 40px;
        text-align: right;
        color: var(--cosmic-blue);
        font-weight: 600;
    }

    .result-error .result-icon {
        color: var(--danger-red);
    }

    .error-message {
        margin: 20px 0;
        text-align: center;
    }

    .error-detail {
        color: var(--danger-red);
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        margin-top: 10px;
    }

    @keyframes quantum-scan {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes data-flow {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    @keyframes progress-scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
`;

// Inject additional styles
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

// Initialize the detector when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.detector = new ExoplanetDetector();
});