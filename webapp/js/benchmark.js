/**
 * AIL-RDD Benchmark & Telemetry Visualizer
 * Renders model cards, architectural characteristics, and comparative metrics table.
 * Strictly zero emojis.
 */

import { state } from './state.js';

const MODEL_SPECS = {
  yolo11s_best: {
    displayName: 'YOLO11s',
    role: 'SOTA Precision',
    description: 'Highest mAP@50 (81.5%). Optimal for high-accuracy highway and urban road inspection.',
  },
  yolov10s_best: {
    displayName: 'YOLOv10s',
    role: 'NMS-Free Architecture',
    description: 'Consistent low-latency detection with dual-label assignments eliminating NMS bottleneck.',
  },
  yolov26n_best: {
    displayName: 'YOLO26n (DGVG)',
    role: 'Ultra-Lightweight',
    description: 'Custom Deformable Convolutions & VoVGSCSP modules. 1.89M parameters, 3.8 MB footprint.',
  },
  yolov8s_best: {
    displayName: 'YOLOv8s',
    role: 'Robust Baseline',
    description: 'Established production baseline with balanced feature pyramid representation.',
  },
};

export class BenchmarkVisualizer {
  constructor(cardsContainer, tableContainer) {
    this.cardsContainer = cardsContainer;
    this.tableContainer = tableContainer;
  }

  render(models) {
    if (!models || models.length === 0) return;

    this.renderCards(models);
    this.renderTable(models);
  }

  renderCards(models) {
    if (!this.cardsContainer) return;
    const activeModelId = state.get('selectedModel');

    this.cardsContainer.innerHTML = models
      .map((m) => {
        const spec = MODEL_SPECS[m.model_name] || {
          displayName: m.model_name,
          role: 'Trained Model',
          description: '',
        };
        const isFeatured = m.model_name === activeModelId;
        const featuredClass = isFeatured ? 'featured' : '';

        return `
        <div class="model-card ${featuredClass}" data-model="${m.model_name}">
          <span class="model-card-badge">${spec.role}</span>
          <h3 class="model-card-title">${spec.displayName}</h3>
          <p class="model-card-desc">${spec.description}</p>
          
          <div class="model-metric-row">
            <span class="model-metric-label">mAP@50</span>
            <span class="model-metric-val">${m.map50}%</span>
          </div>
          <div class="model-metric-row">
            <span class="model-metric-label">mAP@50-95</span>
            <span class="model-metric-val">${m.map50_95}%</span>
          </div>
          <div class="model-metric-row">
            <span class="model-metric-label">Parameters</span>
            <span class="model-metric-val">${m.params_m}M</span>
          </div>
          <div class="model-metric-row">
            <span class="model-metric-label">Compute</span>
            <span class="model-metric-val">${m.gflops} GFLOPs</span>
          </div>
          <div class="model-metric-row">
            <span class="model-metric-label">Model Size</span>
            <span class="model-metric-val">${m.size_mb} MB</span>
          </div>
        </div>
      `;
      })
      .join('');

    // Attach click handlers
    this.cardsContainer.querySelectorAll('.model-card').forEach((card) => {
      card.addEventListener('click', () => {
        const modelId = card.getAttribute('data-model');
        if (modelId) {
          state.set('selectedModel', modelId);
        }
      });
    });
  }

  renderTable(models) {
    if (!this.tableContainer) return;
    const activeModelId = state.get('selectedModel');

    const rowsHtml = models
      .map((m) => {
        const spec = MODEL_SPECS[m.model_name] || { displayName: m.model_name };
        const isHighlight = m.model_name === activeModelId;
        const rowClass = isHighlight ? 'class="highlight-row"' : '';

        return `
        <tr ${rowClass} style="cursor: pointer;" data-model="${m.model_name}">
          <td class="table-model-name">${spec.displayName}</td>
          <td class="table-number"><strong>${m.map50}%</strong></td>
          <td class="table-number">${m.map50_95}%</td>
          <td class="table-number">${m.precision}%</td>
          <td class="table-number">${m.recall}%</td>
          <td class="table-number">${m.params_m}M</td>
          <td class="table-number">${m.gflops}</td>
          <td class="table-number">${m.size_mb} MB</td>
          <td class="table-number">${m.avg_latency_ms} ms</td>
          <td class="table-number">${m.avg_fps}</td>
        </tr>
      `;
      })
      .join('');

    this.tableContainer.innerHTML = `
      <table class="benchmark-table">
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>mAP@50</th>
            <th>mAP@50-95</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>Params</th>
            <th>GFLOPs</th>
            <th>Size</th>
            <th>Avg Latency</th>
            <th>Avg FPS</th>
          </tr>
        </thead>
        <tbody>
          ${rowsHtml}
        </tbody>
      </table>
    `;

    this.tableContainer.querySelectorAll('tbody tr').forEach((row) => {
      row.addEventListener('click', () => {
        const modelId = row.getAttribute('data-model');
        if (modelId) {
          state.set('selectedModel', modelId);
        }
      });
    });
  }
}
