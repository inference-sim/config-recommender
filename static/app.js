// Modern GPU Recommender UI - JavaScript
const API_BASE = '/api';

// State
let currentModels = [];
let currentGPUs = [];
let currentRecommendations = null;
let gpuLibrary = [];
let selectedGPUKeys = [];

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initializeUI();
    await loadGPULibrary();
    await loadModels();
    await loadGPUs();
});

function initializeUI() {
    // Navigation
    initializeNavigation();

    // Input tabs
    initializeInputTabs();

    // Collapsible
    initializeCollapsible();

    // Upload areas
    initializeUploadAreas();

    // Configuration
    initializeConfiguration();

    // Models
    initializeModels();

    // GPUs
    initializeGPUs();

    // Recommendations
    initializeRecommendations();
}

// Navigation
function initializeNavigation() {
    // No special navigation handling needed - only external link now
}

// Input Tabs
function initializeInputTabs() {
    const tabs = document.querySelectorAll('.input-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.target;
            const parent = tab.closest('.card-body');

            // Update tab states
            parent.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update panel visibility
            parent.querySelectorAll('.input-panel').forEach(p => p.classList.remove('active'));
            parent.querySelector(`#${target}`)?.classList.add('active');
        });
    });
}

// Collapsible
function initializeCollapsible() {
    const triggers = document.querySelectorAll('.collapsible-trigger');
    triggers.forEach(trigger => {
        trigger.addEventListener('click', () => {
            const collapsible = trigger.closest('.collapsible');
            collapsible.classList.toggle('open');
        });
    });
}

// Upload Areas
function initializeUploadAreas() {
    // Model JSON upload
    const modelUploadArea = document.getElementById('model-upload-area');
    const modelFileInput = document.getElementById('model-json-file');

    modelUploadArea.addEventListener('click', () => modelFileInput.click());
    modelFileInput.addEventListener('change', handleModelJSONUpload);

    modelUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        modelUploadArea.classList.add('drag-over');
    });

    modelUploadArea.addEventListener('dragleave', () => {
        modelUploadArea.classList.remove('drag-over');
    });

    modelUploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        modelUploadArea.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/json') {
            await processModelJSON(file);
        }
    });

    // GPU JSON upload
    const gpuUploadArea = document.getElementById('gpu-upload-area');
    const gpuFileInput = document.getElementById('gpu-json-file');

    gpuUploadArea.addEventListener('click', () => gpuFileInput.click());
    gpuFileInput.addEventListener('change', handleGPUJSONUpload);

    gpuUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        gpuUploadArea.classList.add('drag-over');
    });

    gpuUploadArea.addEventListener('dragleave', () => {
        gpuUploadArea.classList.remove('drag-over');
    });

    gpuUploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        gpuUploadArea.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/json') {
            await processGPUJSON(file);
        }
    });
}

// Configuration
function initializeConfiguration() {
    const memoryOverhead = document.getElementById('memory-overhead');
    const memoryOverheadValue = document.getElementById('memory-overhead-value');

    memoryOverhead.addEventListener('input', (e) => {
        memoryOverheadValue.textContent = parseFloat(e.target.value).toFixed(2) + 'x';
    });
}

// Models
function initializeModels() {
    document.getElementById('add-model-btn').addEventListener('click', addModel);
    document.getElementById('clear-models-btn').addEventListener('click', clearModels);
}

async function addModel() {
    const name = document.getElementById('model-name').value.trim();
    if (!name) {
        showToast('Please enter a model name', 'error');
        return;
    }

    const modelData = { name };

    const numParams = parseFloat(document.getElementById('model-params').value);
    if (numParams > 0) {
        modelData.num_parameters = numParams;
        modelData.num_layers = parseInt(document.getElementById('model-layers').value);
        modelData.hidden_size = parseInt(document.getElementById('model-hidden').value);
        modelData.num_attention_heads = parseInt(document.getElementById('model-heads').value);
    }

    try {
        const response = await fetch(`${API_BASE}/models`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(modelData)
        });

        const result = await response.json();

        if (response.ok) {
            showToast(result.message, 'success');
            await loadModels();

            // Clear form
            document.getElementById('model-name').value = '';
            document.getElementById('model-params').value = '0';
            document.getElementById('model-layers').value = '0';
            document.getElementById('model-hidden').value = '0';
            document.getElementById('model-heads').value = '0';
        } else {
            showToast(result.detail || 'Error adding model', 'error');
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    }
}

async function handleModelJSONUpload() {
    const file = document.getElementById('model-json-file').files[0];
    if (file) await processModelJSON(file);
}

async function processModelJSON(file) {
    try {
        const text = await file.text();
        const models = JSON.parse(text);

        let successCount = 0;
        for (const modelData of models) {
            try {
                const response = await fetch(`${API_BASE}/models`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(modelData)
                });
                if (response.ok) successCount++;
            } catch (error) {
                console.error('Error adding model:', error);
            }
        }

        showToast(`Loaded ${successCount} models`, 'success');
        await loadModels();
    } catch (error) {
        showToast('Error parsing JSON: ' + error.message, 'error');
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();
        currentModels = data.models;
        renderModelsList();
        updateStats();
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function renderModelsList() {
    const container = document.getElementById('models-list');

    if (currentModels.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                    <circle cx="32" cy="32" r="30" stroke="currentColor" stroke-width="2" stroke-dasharray="4 4"/>
                    <path d="M32 20V44M20 32H44" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <p>No models added yet</p>
            </div>
        `;
        return;
    }

    container.innerHTML = currentModels.map((model, index) => `
        <div class="item-chip">
            <span class="item-chip-label">${model.name}</span>
            <span class="item-chip-detail">${model.num_parameters.toFixed(1)}B</span>
            <button class="item-chip-remove" onclick="deleteModel(${index})">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </button>
        </div>
    `).join('');
}

async function deleteModel(index) {
    try {
        const response = await fetch(`${API_BASE}/models/${index}`, { method: 'DELETE' });
        const result = await response.json();

        if (response.ok) {
            showToast(result.message, 'success');
            await loadModels();
        } else {
            showToast(result.detail || 'Error deleting model', 'error');
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    }
}

async function clearModels() {
    if (currentModels.length === 0) return;
    if (!confirm('Clear all models?')) return;

    try {
        const response = await fetch(`${API_BASE}/models`, { method: 'DELETE' });
        const result = await response.json();

        if (response.ok) {
            showToast(result.message, 'success');
            await loadModels();
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    }
}

// GPUs
function initializeGPUs() {
    document.getElementById('add-library-gpus-btn').addEventListener('click', addGPUsFromLibrary);
    document.getElementById('add-gpu-btn').addEventListener('click', addGPU);
    document.getElementById('clear-gpus-btn').addEventListener('click', clearGPUs);
}

async function loadGPULibrary() {
    try {
        const response = await fetch(`${API_BASE}/gpus/library`);
        const data = await response.json();
        gpuLibrary = data.library;
        renderGPULibrary();
    } catch (error) {
        console.error('Error loading GPU library:', error);
    }
}

function renderGPULibrary() {
    const container = document.getElementById('gpu-library-grid');
    container.innerHTML = gpuLibrary.map(gpu => `
        <div class="gpu-card" data-key="${gpu.key}">
            <div class="gpu-card-name">${gpu.key}</div>
            <div class="gpu-card-specs">${gpu.memory_gb}GB • ${gpu.tflops_fp16} TFLOPS</div>
        </div>
    `).join('');

    // Add click handlers
    container.querySelectorAll('.gpu-card').forEach(card => {
        card.addEventListener('click', () => {
            const key = card.dataset.key;
            if (selectedGPUKeys.includes(key)) {
                selectedGPUKeys = selectedGPUKeys.filter(k => k !== key);
                card.classList.remove('selected');
            } else {
                selectedGPUKeys.push(key);
                card.classList.add('selected');
            }
        });
    });
}

async function addGPUsFromLibrary() {
    if (selectedGPUKeys.length === 0) {
        showToast('Please select at least one GPU', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/gpus/library/add`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(selectedGPUKeys)
        });

        const result = await response.json();

        if (response.ok) {
            showToast(result.message, 'success');
            await loadGPUs();

            // Clear selection
            selectedGPUKeys = [];
            document.querySelectorAll('.gpu-card').forEach(card => card.classList.remove('selected'));
        } else {
            showToast(result.detail || 'Error adding GPUs', 'error');
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    }
}

async function addGPU() {
    const name = document.getElementById('gpu-name').value.trim();
    if (!name) {
        showToast('Please enter a GPU name', 'error');
        return;
    }

    const gpuData = {
        name,
        memory_gb: parseFloat(document.getElementById('gpu-memory').value),
        memory_bandwidth_gb_s: parseFloat(document.getElementById('gpu-bandwidth').value),
        tflops_fp16: parseFloat(document.getElementById('gpu-tflops-fp16').value),
        tflops_fp32: parseFloat(document.getElementById('gpu-tflops-fp32').value),
        cost_per_hour: parseFloat(document.getElementById('gpu-cost').value)
    };

    try {
        const response = await fetch(`${API_BASE}/gpus`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(gpuData)
        });

        const result = await response.json();

        if (response.ok) {
            showToast(result.message, 'success');
            await loadGPUs();
            document.getElementById('gpu-name').value = '';
        } else {
            showToast(result.detail || 'Error adding GPU', 'error');
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    }
}

async function handleGPUJSONUpload() {
    const file = document.getElementById('gpu-json-file').files[0];
    if (file) await processGPUJSON(file);
}

async function processGPUJSON(file) {
    try {
        const text = await file.text();
        const gpus = JSON.parse(text);

        let successCount = 0;
        for (const gpuData of gpus) {
            try {
                const response = await fetch(`${API_BASE}/gpus`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(gpuData)
                });
                if (response.ok) successCount++;
            } catch (error) {
                console.error('Error adding GPU:', error);
            }
        }

        showToast(`Loaded ${successCount} GPUs`, 'success');
        await loadGPUs();
    } catch (error) {
        showToast('Error parsing JSON: ' + error.message, 'error');
    }
}

async function loadGPUs() {
    try {
        const response = await fetch(`${API_BASE}/gpus`);
        const data = await response.json();
        currentGPUs = data.gpus;
        renderGPUsList();
        updateStats();
    } catch (error) {
        console.error('Error loading GPUs:', error);
    }
}

function renderGPUsList() {
    const container = document.getElementById('gpus-list');

    if (currentGPUs.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                    <circle cx="32" cy="32" r="30" stroke="currentColor" stroke-width="2" stroke-dasharray="4 4"/>
                    <path d="M32 20V44M20 32H44" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <p>No GPUs added yet</p>
            </div>
        `;
        return;
    }

    container.innerHTML = currentGPUs.map((gpu, index) => `
        <div class="item-chip">
            <span class="item-chip-label">${gpu.name}</span>
            <span class="item-chip-detail">${gpu.memory_gb}GB</span>
            <button class="item-chip-remove" onclick="deleteGPU(${index})">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </button>
        </div>
    `).join('');
}

async function deleteGPU(index) {
    try {
        const response = await fetch(`${API_BASE}/gpus/${index}`, { method: 'DELETE' });
        const result = await response.json();

        if (response.ok) {
            showToast(result.message, 'success');
            await loadGPUs();
        } else {
            showToast(result.detail || 'Error deleting GPU', 'error');
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    }
}

async function clearGPUs() {
    if (currentGPUs.length === 0) return;
    if (!confirm('Clear all GPUs?')) return;

    try {
        const response = await fetch(`${API_BASE}/gpus`, { method: 'DELETE' });
        const result = await response.json();

        if (response.ok) {
            showToast(result.message, 'success');
            await loadGPUs();
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    }
}

// Recommendations
function initializeRecommendations() {
    document.getElementById('generate-recommendations').addEventListener('click', generateRecommendations);
    document.getElementById('export-json').addEventListener('click', exportJSON);
    document.getElementById('export-csv').addEventListener('click', exportCSV);
}

async function generateRecommendations() {
    if (currentModels.length === 0 || currentGPUs.length === 0) {
        document.getElementById('recommendation-warning').classList.remove('hidden');
        return;
    }

    document.getElementById('recommendation-warning').classList.add('hidden');
    document.getElementById('recommendations-loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');

    const precision = document.querySelector('input[name="precision"]:checked').value;
    const inputLength = parseInt(document.getElementById('input-length').value);
    const outputLength = parseInt(document.getElementById('output-length').value);
    const memoryOverhead = parseFloat(document.getElementById('memory-overhead').value);
    const latencyBound = parseFloat(document.getElementById('latency-bound').value) || null;

    const requestData = {
        models: currentModels.map(m => ({
            name: m.name,
            num_parameters: m.num_parameters || null,
            num_layers: m.num_layers || null,
            hidden_size: m.hidden_size || null,
            num_attention_heads: m.num_attention_heads || null
        })),
        gpus: currentGPUs,
        precision,
        input_length: inputLength > 0 ? inputLength : null,
        output_length: outputLength > 0 ? outputLength : null,
        memory_overhead: memoryOverhead,
        latency_bound_ms: latencyBound
    };

    try {
        const response = await fetch(`${API_BASE}/recommendations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();

        if (response.ok) {
            currentRecommendations = result;
            renderRecommendations();
            showToast('Recommendations generated', 'success');

            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        } else {
            showToast(result.detail || 'Error generating recommendations', 'error');
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    } finally {
        document.getElementById('recommendations-loading').classList.add('hidden');
    }
}

function renderRecommendations() {
    if (!currentRecommendations) return;

    document.getElementById('results').classList.remove('hidden');

    const recommendations = currentRecommendations.recommendations;

    // Render each recommendation with visualizations
    const summaryHTML = recommendations.map(rec => {
        if (!rec.performance) {
            return `
                <div class="recommendation-card">
                    <div class="recommendation-header">
                        <div class="recommendation-model">${rec.model_name}</div>
                        <div class="recommendation-gpu">No compatible GPU</div>
                    </div>
                    <div class="reasoning-box">${rec.reasoning}</div>
                </div>
            `;
        }

        // Find the GPU details
        const gpu = currentGPUs.find(g => g.name === rec.recommended_gpu);
        const gpuMemory = gpu ? gpu.memory_gb : 100;

        // Calculate percentages for visualizations
        const memoryUsedPercent = (rec.performance.memory_required_gb / gpuMemory) * 100;
        const weightsPercent = (rec.performance.memory_weights_gb / gpuMemory) * 100;
        const kvCachePercent = (rec.performance.memory_kv_cache_gb / gpuMemory) * 100;
        const availablePercent = 100 - memoryUsedPercent;

        // Find max throughput for comparison bars
        const allThroughputs = rec.all_compatible_gpus.map(g => g.tokens_per_second || 0);
        const maxThroughput = Math.max(...allThroughputs, rec.performance.tokens_per_second);

        return `
            <div class="recommendation-card">
                <div class="recommendation-header">
                    <div class="recommendation-model">${rec.model_name}</div>
                    <div class="recommendation-gpu">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="2" y="3" width="20" height="14" rx="2"/>
                            <line x1="8" y1="21" x2="16" y2="21"/>
                            <line x1="12" y1="17" x2="12" y2="21"/>
                        </svg>
                        ${rec.recommended_gpu}
                    </div>
                </div>

                <!-- Visual Performance Metrics -->
                <div class="visual-metrics">
                    <div class="metric-visual">
                        <div class="metric-visual-header">
                            <span class="metric-visual-label">Throughput</span>
                            <span class="metric-visual-value">
                                ${rec.performance.tokens_per_second.toFixed(1)}
                                <span class="metric-visual-unit">tok/s</span>
                            </span>
                        </div>
                        <div class="metric-visual-subtitle">Higher is better</div>
                    </div>

                    <div class="metric-visual">
                        <div class="metric-visual-header">
                            <span class="metric-visual-label">Latency</span>
                            <span class="metric-visual-value">
                                ${rec.performance.intertoken_latency_ms.toFixed(2)}
                                <span class="metric-visual-unit">ms</span>
                            </span>
                        </div>
                        <div class="metric-visual-subtitle">Lower is better</div>
                    </div>

                    <div class="metric-visual">
                        <div class="metric-visual-header">
                            <span class="metric-visual-label">Memory Usage</span>
                            <span class="metric-visual-value">
                                ${rec.performance.memory_required_gb.toFixed(1)}
                                <span class="metric-visual-unit">GB</span>
                            </span>
                        </div>
                        <div class="metric-visual-subtitle">${memoryUsedPercent.toFixed(1)}% of ${gpuMemory}GB</div>
                    </div>

                    <div class="metric-visual">
                        <div class="metric-visual-header">
                            <span class="metric-visual-label">TP Size</span>
                            <span class="metric-visual-value">
                                ${rec.performance.tensor_parallel_size}
                                <span class="metric-visual-unit">${rec.performance.tensor_parallel_size === 1 ? 'GPU' : 'GPUs'}</span>
                            </span>
                        </div>
                        <div class="metric-visual-subtitle">Tensor parallelism degree</div>
                    </div>
                </div>

                <!-- Memory Breakdown -->
                <div class="memory-chart">
                    <div class="memory-chart-title">📊 Memory Breakdown</div>
                    <div class="memory-chart-content">
                        <div class="memory-pie-chart" style="background: conic-gradient(
                            from 0deg,
                            #667eea 0% ${weightsPercent}%,
                            #48bb78 ${weightsPercent}% ${weightsPercent + kvCachePercent}%,
                            #e2e8f0 ${weightsPercent + kvCachePercent}% 100%
                        );"></div>
                        <div class="memory-legend">
                            <div class="legend-item">
                                <div class="legend-color weights"></div>
                                <div class="legend-label">
                                    <div class="legend-name">Weights</div>
                                    <div class="legend-value">${rec.performance.memory_weights_gb.toFixed(1)} GB (${weightsPercent.toFixed(1)}%)</div>
                                </div>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color kv-cache"></div>
                                <div class="legend-label">
                                    <div class="legend-name">KV Cache</div>
                                    <div class="legend-value">${rec.performance.memory_kv_cache_gb.toFixed(1)} GB (${kvCachePercent.toFixed(1)}%)</div>
                                </div>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color available"></div>
                                <div class="legend-label">
                                    <div class="legend-name">Available</div>
                                    <div class="legend-value">${(gpuMemory - rec.performance.memory_required_gb).toFixed(1)} GB (${availablePercent.toFixed(1)}%)</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- GPU Comparison -->
                ${rec.all_compatible_gpus && rec.all_compatible_gpus.length > 0 ? `
                    <div class="comparison-section">
                        <div class="comparison-title">⚖️ All Compatible GPUs</div>
                        <div class="comparison-chart">
                            <div class="comparison-bars">
                                ${rec.all_compatible_gpus.slice(0, 5).map(gpu => {
                                    const heightPercent = (gpu.tokens_per_second / maxThroughput) * 100;
                                    return `
                                        <div class="comparison-bar-item">
                                            <div class="comparison-bar-container">
                                                <div class="comparison-bar-fill ${gpu.gpu_name === rec.recommended_gpu ? 'recommended' : ''}"
                                                     style="height: ${heightPercent}%">
                                                </div>
                                            </div>
                                            <div class="comparison-bar-label">
                                                <div class="comparison-bar-name">${gpu.gpu_name}</div>
                                                <div class="comparison-bar-value">${gpu.tokens_per_second?.toFixed(1) || 'N/A'} tok/s</div>
                                                ${gpu.gpu_name === rec.recommended_gpu ? '<div class="performance-badge">✓ Best</div>' : ''}
                                            </div>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>
                    </div>
                ` : ''}

                <!-- Reasoning -->
                <div class="reasoning-box" style="margin-top: 1.5rem;">
                    <strong>💡 Reasoning:</strong> ${rec.reasoning}
                </div>
            </div>
        `;
    }).join('');

    document.getElementById('recommendations-summary').innerHTML = summaryHTML;

    // Clear details section as we now show everything in summary
    document.getElementById('recommendations-details').innerHTML = '';

    updateStats();
}

function exportJSON() {
    if (!currentRecommendations) return;

    const dataStr = JSON.stringify(currentRecommendations, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gpu_recommendations.json';
    a.click();
    URL.revokeObjectURL(url);

    showToast('JSON exported successfully', 'success');
}

function exportCSV() {
    if (!currentRecommendations) return;

    const recommendations = currentRecommendations.recommendations;

    let csv = 'Model,Recommended GPU,Throughput (tok/s),Latency (ms/token),Memory (GB),TP Size\n';

    recommendations.forEach(rec => {
        csv += `"${rec.model_name}",`;
        csv += `"${rec.recommended_gpu || 'None'}",`;
        csv += `${rec.performance ? rec.performance.tokens_per_second.toFixed(1) : 'N/A'},`;
        csv += `${rec.performance ? rec.performance.intertoken_latency_ms.toFixed(2) : 'N/A'},`;
        csv += `${rec.performance ? rec.performance.memory_required_gb.toFixed(1) : 'N/A'},`;
        csv += `${rec.performance ? rec.performance.tensor_parallel_size : 'N/A'}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gpu_recommendations.csv';
    a.click();
    URL.revokeObjectURL(url);

    showToast('CSV exported successfully', 'success');
}

// Stats
function updateStats() {
    document.getElementById('models-count').textContent = currentModels.length;
    document.getElementById('gpus-count').textContent = currentGPUs.length;
    document.getElementById('recommendations-count').textContent =
        currentRecommendations ? currentRecommendations.recommendations.length : 0;
}

// Toast Notifications
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = document.createElement('div');
    icon.innerHTML = type === 'success'
        ? '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"/></svg>'
        : '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"/></svg>';

    const messageEl = document.createElement('div');
    messageEl.className = 'toast-message';
    messageEl.textContent = message;

    toast.appendChild(icon);
    toast.appendChild(messageEl);
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
