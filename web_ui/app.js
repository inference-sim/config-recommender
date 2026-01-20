// Configuration
const API_BASE_URL = 'http://localhost:8000';

// State
let models = [];
let gpus = [];
let selectedGPUKeys = new Set();
let recommendations = null;

// GPU Library Data
const GPU_LIBRARY = {
    'H100': { name: 'NVIDIA H100 80GB', memory_gb: 80, memory_bandwidth_gb_s: 3350, tflops_fp16: 989, tflops_fp32: 67, cost_per_hour: 8.00 },
    'H200': { name: 'NVIDIA H200 141GB', memory_gb: 141, memory_bandwidth_gb_s: 4800, tflops_fp16: 989, tflops_fp32: 67, cost_per_hour: 12.00 },
    'A100-80GB': { name: 'NVIDIA A100 80GB', memory_gb: 80, memory_bandwidth_gb_s: 2039, tflops_fp16: 312, tflops_fp32: 156, cost_per_hour: 3.67 },
    'A100-40GB': { name: 'NVIDIA A100 40GB', memory_gb: 40, memory_bandwidth_gb_s: 1555, tflops_fp16: 312, tflops_fp32: 156, cost_per_hour: 2.50 },
    'L40': { name: 'NVIDIA L40 48GB', memory_gb: 48, memory_bandwidth_gb_s: 864, tflops_fp16: 362, tflops_fp32: 181, cost_per_hour: 2.00 },
    'L4': { name: 'NVIDIA L4 24GB', memory_gb: 24, memory_bandwidth_gb_s: 300, tflops_fp16: 121, tflops_fp32: 60, cost_per_hour: 1.00 },
    'V100': { name: 'NVIDIA V100 32GB', memory_gb: 32, memory_bandwidth_gb_s: 900, tflops_fp16: 125, tflops_fp32: 15.7, cost_per_hour: 2.50 },
    'T4': { name: 'NVIDIA T4 16GB', memory_gb: 16, memory_bandwidth_gb_s: 320, tflops_fp16: 65, tflops_fp32: 8.1, cost_per_hour: 0.50 }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadFromLocalStorage();
    updateUI();
    checkBackendStatus();

    // File upload handlers
    document.getElementById('modelsFile').addEventListener('change', uploadModels);
});

// Theme Toggle
document.getElementById('themeToggle').addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    document.querySelector('.theme-icon').textContent = isDark ? '☀️' : '🌙';
    localStorage.setItem('darkMode', isDark);
});

// Load dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
    document.querySelector('.theme-icon').textContent = '☀️';
}

// Models Management
function addModel() {
    const input = document.getElementById('modelInput');
    const modelName = input.value.trim();

    if (!modelName) {
        showToast('Please enter a model name', 'error');
        return;
    }

    if (models.some(m => m.name === modelName)) {
        showToast('Model already added', 'warning');
        return;
    }

    models.push({ name: modelName });
    input.value = '';
    saveToLocalStorage();
    updateUI();
    showToast('Model added successfully', 'success');
}

function removeModel(index) {
    models.splice(index, 1);
    saveToLocalStorage();
    updateUI();
    showToast('Model removed', 'success');
}

function clearModels() {
    if (models.length === 0) return;
    if (confirm('Clear all models?')) {
        models = [];
        saveToLocalStorage();
        updateUI();
        showToast('All models cleared', 'success');
    }
}

function uploadModels(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);
            if (Array.isArray(data)) {
                let added = 0;
                data.forEach(model => {
                    if (model.name && !models.some(m => m.name === model.name)) {
                        models.push(model);
                        added++;
                    }
                });
                saveToLocalStorage();
                updateUI();
                showToast(`Loaded ${added} models`, 'success');
            }
        } catch (error) {
            showToast('Error parsing JSON: ' + error.message, 'error');
        }
    };
    reader.readAsText(file);
}

// GPUs Management
function toggleGPU(key) {
    const card = event.currentTarget;

    if (selectedGPUKeys.has(key)) {
        selectedGPUKeys.delete(key);
        card.classList.remove('selected');
        // Remove from gpus array
        gpus = gpus.filter(g => !g.name.includes(GPU_LIBRARY[key].name));
    } else {
        selectedGPUKeys.add(key);
        card.classList.add('selected');
        // Add to gpus array
        const gpuData = GPU_LIBRARY[key];
        if (!gpus.some(g => g.name === gpuData.name)) {
            gpus.push({ ...gpuData });
        }
    }

    saveToLocalStorage();
    updateUI();
}

function removeGPU(index) {
    const gpu = gpus[index];
    gpus.splice(index, 1);

    // Update selected keys
    for (const [key, data] of Object.entries(GPU_LIBRARY)) {
        if (data.name === gpu.name) {
            selectedGPUKeys.delete(key);
            const card = document.querySelector(`.gpu-card[data-gpu="${key}"]`);
            if (card) card.classList.remove('selected');
            break;
        }
    }

    saveToLocalStorage();
    updateUI();
    showToast('GPU removed', 'success');
}

function clearGPUs() {
    if (gpus.length === 0) return;
    if (confirm('Clear all GPUs?')) {
        gpus = [];
        selectedGPUKeys.clear();
        document.querySelectorAll('.gpu-card').forEach(card => {
            card.classList.remove('selected');
        });
        saveToLocalStorage();
        updateUI();
        showToast('All GPUs cleared', 'success');
    }
}

// Generate Recommendations
async function generateRecommendations() {
    if (models.length === 0) {
        showToast('Please add at least one model', 'error');
        return;
    }

    if (gpus.length === 0) {
        showToast('Please select at least one GPU', 'error');
        return;
    }

    // Show results section and loading state
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    document.getElementById('loadingState').style.display = 'block';
    document.getElementById('resultsState').style.display = 'none';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Get configuration
    const precision = parseInt(document.getElementById('precision').value);
    const inputLength = parseInt(document.getElementById('inputLength').value) || null;
    const outputLength = parseInt(document.getElementById('outputLength').value) || null;
    const memoryOverhead = parseFloat(document.getElementById('memoryOverhead').value);
    const latencyBound = parseFloat(document.getElementById('latencyBound').value) || null;

    const sequenceLength = (inputLength && outputLength) ? inputLength + outputLength : null;

    const requestData = {
        model_names: models.map(m => m.name),
        gpu_names: gpus.map(g => g.name),
        precision_bytes: precision,
        memory_overhead_factor: memoryOverhead,
        latency_bound_ms: latencyBound,
        input_length: inputLength,
        output_length: outputLength,
        sequence_length: sequenceLength
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/recommendations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`HTTP ${response.status}: ${error}`);
        }

        const data = await response.json();
        recommendations = data.recommendations;
        displayRecommendations();
        showToast('Recommendations generated successfully', 'success');
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
        console.error('Error:', error);
        resultsSection.style.display = 'none';
    } finally {
        document.getElementById('loadingState').style.display = 'none';
    }
}

function displayRecommendations() {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';

    document.getElementById('resultsState').style.display = 'block';

    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No recommendations available</p></div>';
        return;
    }

    recommendations.forEach(rec => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const hasGPU = rec.recommended_gpu !== null;
        const badge = hasGPU ?
            '<span class="result-badge badge-success">✓ Compatible</span>' :
            '<span class="result-badge badge-warning">⚠ No Compatible GPU</span>';

        let metricsHTML = '';
        if (rec.performance) {
            const perf = rec.performance;
            metricsHTML = `
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-label">Throughput</div>
                        <div class="metric-value">${perf.tokens_per_second ? perf.tokens_per_second.toFixed(1) : 'N/A'}</div>
                        <div class="metric-label">tokens/sec</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Latency</div>
                        <div class="metric-value">${perf.intertoken_latency_ms ? perf.intertoken_latency_ms.toFixed(2) : 'N/A'}</div>
                        <div class="metric-label">ms/token</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Memory</div>
                        <div class="metric-value">${perf.memory_required_gb ? perf.memory_required_gb.toFixed(1) : 'N/A'}</div>
                        <div class="metric-label">GB used</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">TP Size</div>
                        <div class="metric-value">${perf.tensor_parallel_size || 1}</div>
                        <div class="metric-label">GPUs</div>
                    </div>
                </div>
            `;
        }

        card.innerHTML = `
            <div class="result-header">
                <div class="result-title">${rec.model_name}</div>
                ${badge}
            </div>
            ${hasGPU ? `<h3 style="color: var(--primary); margin-bottom: 1rem;">→ ${rec.recommended_gpu}</h3>` : ''}
            ${metricsHTML}
            <div class="reasoning-box">
                <strong>Analysis:</strong>
                <p>${rec.reasoning}</p>
            </div>
        `;

        container.appendChild(card);
    });
}

// Export Functions
function exportJSON() {
    if (!recommendations) {
        showToast('No recommendations to export', 'error');
        return;
    }

    const dataStr = JSON.stringify({ recommendations }, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'gpu_recommendations.json';
    link.click();
    URL.revokeObjectURL(url);
    showToast('JSON exported successfully', 'success');
}

function exportCSV() {
    if (!recommendations) {
        showToast('No recommendations to export', 'error');
        return;
    }

    let csv = 'Model,Recommended GPU,Throughput (tok/s),Latency (ms),Memory (GB),TP Size\n';

    recommendations.forEach(rec => {
        const perf = rec.performance;
        csv += `"${rec.model_name}","${rec.recommended_gpu || 'None'}",`;
        if (perf) {
            csv += `${perf.tokens_per_second || 'N/A'},${perf.intertoken_latency_ms || 'N/A'},`;
            csv += `${perf.memory_required_gb || 'N/A'},${perf.tensor_parallel_size || 1}\n`;
        } else {
            csv += 'N/A,N/A,N/A,N/A\n';
        }
    });

    const dataBlob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'gpu_recommendations.csv';
    link.click();
    URL.revokeObjectURL(url);
    showToast('CSV exported successfully', 'success');
}

// UI Updates
function updateUI() {
    updateModelsList();
    updateGPUsList();
    updateGenerateButton();
}

function updateModelsList() {
    const container = document.getElementById('modelsList');
    const section = document.getElementById('modelsListSection');
    const count = document.getElementById('modelCount');

    count.textContent = models.length;

    if (models.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    container.innerHTML = '';

    models.forEach((model, index) => {
        const chip = document.createElement('div');
        chip.className = 'item-chip';
        chip.innerHTML = `
            <div class="item-name">📄 ${model.name}</div>
            <button class="item-remove" onclick="removeModel(${index})">×</button>
        `;
        container.appendChild(chip);
    });
}

function updateGPUsList() {
    const container = document.getElementById('gpusList');
    const section = document.getElementById('gpusListSection');
    const count = document.getElementById('gpuCount');

    count.textContent = gpus.length;

    if (gpus.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    container.innerHTML = '';

    gpus.forEach((gpu, index) => {
        const chip = document.createElement('div');
        chip.className = 'item-chip';
        chip.innerHTML = `
            <div class="item-name">🖥️ ${gpu.name}</div>
            <button class="item-remove" onclick="removeGPU(${index})">×</button>
        `;
        container.appendChild(chip);
    });
}

function updateGenerateButton() {
    const btn = document.getElementById('generateBtn');
    if (btn) {
        btn.disabled = models.length === 0 || gpus.length === 0;
    }
}

// Local Storage
function saveToLocalStorage() {
    localStorage.setItem('models', JSON.stringify(models));
    localStorage.setItem('gpus', JSON.stringify(gpus));
    localStorage.setItem('selectedGPUKeys', JSON.stringify([...selectedGPUKeys]));
}

function loadFromLocalStorage() {
    const savedModels = localStorage.getItem('models');
    const savedGPUs = localStorage.getItem('gpus');
    const savedKeys = localStorage.getItem('selectedGPUKeys');

    if (savedModels) {
        try {
            models = JSON.parse(savedModels);
        } catch (e) {
            console.error('Error loading models:', e);
        }
    }

    if (savedGPUs) {
        try {
            gpus = JSON.parse(savedGPUs);
        } catch (e) {
            console.error('Error loading GPUs:', e);
        }
    }

    if (savedKeys) {
        try {
            selectedGPUKeys = new Set(JSON.parse(savedKeys));
            // Update UI to show selected GPUs
            setTimeout(() => {
                selectedGPUKeys.forEach(key => {
                    const card = document.querySelector(`.gpu-card[data-gpu="${key}"]`);
                    if (card) {
                        card.classList.add('selected');
                    }
                });
            }, 100);
        } catch (e) {
            console.error('Error loading GPU keys:', e);
        }
    }
}

// Toast Notifications
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Check Backend Status
async function checkBackendStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (response.ok) {
            console.log('✓ Backend is running');
        }
    } catch (error) {
        showToast('Warning: Backend not running', 'warning');
        console.warn('Backend not available:', error);
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const modelInput = document.getElementById('modelInput');
        if (document.activeElement === modelInput) {
            addModel();
        }
    }
});

// Made with Bob
