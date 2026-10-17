/**
 * AIL-RDD Main Application Coordinator
 * Coordinates Sidebar Navigation, Model Switcher, 5-Column Dataset Gallery,
 * High-DPI Canvas Viewport, and Live Telemetry.
 * Strictly zero emojis.
 */

import { state } from './state.js';
import { api } from './api.js';
import { DetectionCanvas } from './canvas.js';
import { BenchmarkVisualizer } from './benchmark.js';

class App {
  constructor() {
    this.canvasEngine = null;
    this.benchmarkVisualizer = null;
    this.showDetections = true;
    this._toastTimer = null;
    this.BATCH_SIZE = 10;
  }

  async init() {
    // 1. Initialize Canvas
    const canvasEl = document.getElementById('detectionCanvas');
    const containerEl = document.getElementById('canvasContainer');
    this.canvasEngine = new DetectionCanvas(canvasEl, containerEl);
    this.canvasEngine.resize();

    // 2. Initialize Benchmark Visualizer
    const cardsEl = document.getElementById('modelsCardsContainer');
    const tableEl = document.getElementById('benchmarkTableContainer');
    this.benchmarkVisualizer = new BenchmarkVisualizer(cardsEl, tableEl);

    // 3. Bind UI Events
    this._bindViewSwitcher();
    this._bindModelSwitcher();
    this._bindSliders();
    this._bindControls();
    this._bindDropzone();
    this._bindDatasetPagination();
    this._bindStateSubscribers();

    // 4. Check Backend Status & Load Initial Models & Benchmark
    await this._loadBackendAndData();

    // 5. Load Initial Batch of 10 Dataset Images
    await this._loadDatasetBatch(true);
  }

  async _loadBackendAndData() {
    const status = await api.checkStatus();
    state.set('isBackendLive', status.isLive);

    const statusDot = document.getElementById('backendStatusDot');
    const statusText = document.getElementById('backendStatusText');

    if (status.isLive) {
      statusDot?.classList.remove('offline');
      if (statusText) statusText.textContent = `Live (${status.data.device.toUpperCase()})`;
    } else {
      statusDot?.classList.add('offline');
      if (statusText) statusText.textContent = 'Standalone Mode';
    }

    // Load models
    const models = await api.fetchModels();
    state.set('models', models);
    this.benchmarkVisualizer.render(models);
  }

  _bindViewSwitcher() {
    const navBtns = document.querySelectorAll('.sidebar-nav-btn');
    navBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const viewId = btn.getAttribute('data-view');
        if (!viewId) return;

        navBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.view-pane').forEach((pane) => {
          pane.classList.remove('active');
        });

        const targetPane = document.getElementById(`view${viewId.charAt(0).toUpperCase() + viewId.slice(1)}`);
        targetPane?.classList.add('active');

        state.set('activeView', viewId);

        if (viewId === 'studio') {
          setTimeout(() => this.canvasEngine.resize(), 50);
        }
      });
    });
  }

  _bindModelSwitcher() {
    const modelBtns = document.querySelectorAll('.sidebar-model-btn');
    modelBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const modelId = btn.getAttribute('data-model');
        if (modelId) {
          state.set('selectedModel', modelId);
          this._showToast(`Switched model to ${this._getModelDisplayName(modelId)}`);
        }
      });
    });
  }

  _bindSliders() {
    const confSlider = document.getElementById('confSlider');
    const confVal = document.getElementById('confSliderVal');
    const iouSlider = document.getElementById('iouSlider');
    const iouVal = document.getElementById('iouSliderVal');

    let confDebounceTimer = null;
    let iouDebounceTimer = null;

    if (confSlider && confVal) {
      confSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        confVal.textContent = val.toFixed(2);
        clearTimeout(confDebounceTimer);
        confDebounceTimer = setTimeout(() => {
          state.set('confidenceThreshold', val);
        }, 160);
      });
    }

    if (iouSlider && iouVal) {
      iouSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        iouVal.textContent = val.toFixed(2);
        clearTimeout(iouDebounceTimer);
        iouDebounceTimer = setTimeout(() => {
          state.set('iouThreshold', val);
        }, 160);
      });
    }
  }

  _bindControls() {
    // Reset View
    const btnResetView = document.getElementById('btnResetView');
    btnResetView?.addEventListener('click', () => {
      this.canvasEngine.resetTransform();
    });

    // Toggle Bounding Boxes
    const btnToggleBoxes = document.getElementById('btnToggleBoxes');
    btnToggleBoxes?.addEventListener('click', () => {
      this.showDetections = !this.showDetections;
      btnToggleBoxes.classList.toggle('active', this.showDetections);
      const activeDets = state.get('detections');
      this.canvasEngine.setDetections(this.showDetections ? activeDets : []);
    });

    // Upload Button
    const fileInput = document.getElementById('fileInput');
    const btnUpload = document.getElementById('btnUpload');

    btnUpload?.addEventListener('click', () => fileInput?.click());
    fileInput?.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) this._handleFileUpload(file);
    });

    // Run Detection Manual Button
    const btnRunDetect = document.getElementById('btnRunDetect');
    btnRunDetect?.addEventListener('click', () => {
      this._triggerInference();
    });

    // Damage Class Filters in Telemetry
    document.querySelectorAll('.damage-item').forEach((item) => {
      item.addEventListener('click', () => {
        const cls = item.getAttribute('data-class');
        const curr = state.get('activeFilter');
        const next = curr === cls ? 'all' : cls;
        state.set('activeFilter', next);
      });
    });
  }

  _bindDropzone() {
    const container = document.getElementById('canvasContainer');
    if (!container) return;

    ['dragenter', 'dragover'].forEach((eventName) => {
      container.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        container.style.outline = '2px dashed var(--color-primary)';
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      container.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        container.style.outline = 'none';
      });
    });

    container.addEventListener('drop', (e) => {
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        this._handleFileUpload(files[0]);
      }
    });
  }

  _bindDatasetPagination() {
    const btnLoadMore = document.getElementById('btnLoadMore');
    btnLoadMore?.addEventListener('click', async () => {
      if (state.get('isLoadingMore')) return;
      await this._loadDatasetBatch(false);
    });
  }

  async _loadDatasetBatch(isInitial = false) {
    state.set('isLoadingMore', true);
    const btnLoadMore = document.getElementById('btnLoadMore');
    if (btnLoadMore) {
      btnLoadMore.disabled = true;
      btnLoadMore.querySelector('span').textContent = 'Đang tải...';
    }

    const currentOffset = isInitial ? 0 : state.get('datasetOffset');
    try {
      const res = await api.fetchDatasetImages(currentOffset, this.BATCH_SIZE);
      const incoming = res.images || [];
      const total = res.total || 3543;

      let allImages = [];
      if (isInitial) {
        allImages = incoming;
      } else {
        const existing = state.get('datasetImages') || [];
        allImages = [...existing, ...incoming];
      }

      state.set('datasetImages', allImages);
      state.set('datasetOffset', currentOffset + incoming.length);
      state.set('datasetTotal', total);

      // Render 5-column grid
      this._renderDatasetGrid(allImages, incoming, isInitial);

      // Update counter
      const counterEl = document.getElementById('datasetCounter');
      if (counterEl) {
        counterEl.textContent = `Showing ${allImages.length} of ${total.toLocaleString()} images`;
      }

      // Check if more available
      if (btnLoadMore) {
        if (!res.has_more || allImages.length >= total) {
          btnLoadMore.style.display = 'none';
        } else {
          btnLoadMore.style.display = 'flex';
          btnLoadMore.disabled = false;
          btnLoadMore.querySelector('span').textContent = 'Tải thêm (+10)';
        }
      }

      // If initial, select the first image
      if (isInitial && allImages.length > 0) {
        this._selectDatasetImage(allImages[0]);
      }
    } catch (err) {
      console.error('Failed to load dataset batch:', err);
      if (btnLoadMore) {
        btnLoadMore.disabled = false;
        btnLoadMore.querySelector('span').textContent = 'Thử lại';
      }
    } finally {
      state.set('isLoadingMore', false);
    }
  }

  _renderDatasetGrid(allImages, newBatch, isInitial) {
    const gridEl = document.getElementById('datasetGrid5');
    if (!gridEl) return;

    if (isInitial) {
      gridEl.innerHTML = '';
    }

    const itemsToAppend = isInitial ? allImages : newBatch;
    const activeImage = state.get('activeImage');

    itemsToAppend.forEach((imgItem) => {
      const card = document.createElement('div');
      card.className = 'gallery-item-card';
      card.setAttribute('data-id', imgItem.id);
      card.setAttribute('data-url', imgItem.url);

      if (activeImage && (activeImage.id === imgItem.id || activeImage.id === imgItem.filename)) {
        card.classList.add('active');
      }

      card.innerHTML = `
        <div class="gallery-thumb-wrap">
          <img class="gallery-thumb-img" src="${imgItem.url}" alt="${imgItem.title}" loading="lazy" />
        </div>
        <div class="gallery-item-info">
          <span class="gallery-item-title" title="${imgItem.filename}">${imgItem.filename}</span>
          <span class="gallery-item-tag">RDD</span>
        </div>
      `;

      card.addEventListener('click', () => {
        this._selectDatasetImage(imgItem);
      });

      gridEl.appendChild(card);
    });
  }

  _selectDatasetImage(imgItem) {
    document.querySelectorAll('.gallery-item-card').forEach((card) => {
      const isMatch = card.getAttribute('data-id') === imgItem.id;
      card.classList.toggle('active', isMatch);
    });

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      state.set('activeImage', {
        id: imgItem.id,
        file: imgItem.filename,
        url: imgItem.url,
        width: img.naturalWidth,
        height: img.naturalHeight,
        element: img,
      });

      this._updateStageMeta(imgItem.filename, `${img.naturalWidth} x ${img.naturalHeight}px`);
      this._triggerInference();
    };
    img.src = imgItem.url;
  }

  _handleFileUpload(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPEG, PNG).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        state.set('activeImage', {
          id: file.name,
          file: file.name,
          url: e.target.result,
          width: img.naturalWidth,
          height: img.naturalHeight,
          element: img,
          base64: e.target.result,
        });

        // De-select gallery items
        document.querySelectorAll('.gallery-item-card').forEach((c) => c.classList.remove('active'));

        this._updateStageMeta(file.name, `${img.naturalWidth} x ${img.naturalHeight}px`);
        this._showToast(`Loaded ${file.name}`);
        this._triggerInference();
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  async _triggerInference() {
    const activeImage = state.get('activeImage');
    if (!activeImage) return;

    state.set('isDetecting', true);

    const model = state.get('selectedModel');
    const conf = state.get('confidenceThreshold');
    const iou = state.get('iouThreshold');

    try {
      const result = await api.detect({
        model,
        conf,
        iou,
        sampleId: activeImage.file || activeImage.id,
        imageBase64: activeImage.base64,
      });

      if (!result) return; // Aborted by newer request

      state.set('detections', result.detections || []);
      state.set('metrics', {
        latency_ms: result.latency_ms || 0,
        count: result.count || 0,
        speed: result.speed || { preprocess_ms: 0, inference_ms: 0, postprocess_ms: 0 },
      });
    } catch (err) {
      console.error('Detection request failed:', err);
    } finally {
      state.set('isDetecting', false);
    }
  }

  _bindStateSubscribers() {
    // Model change
    state.subscribe('selectedModel', (modelId) => {
      document.querySelectorAll('.sidebar-model-btn').forEach((btn) => {
        const isSelected = btn.getAttribute('data-model') === modelId;
        btn.classList.toggle('active', isSelected);
        btn.setAttribute('aria-selected', isSelected ? 'true' : 'false');
      });
      const models = state.get('models');
      this.benchmarkVisualizer.render(models);
      this._triggerInference();
    });

    // Confidence / IOU changes
    state.subscribe('confidenceThreshold', () => {
      this._triggerInference();
    });

    state.subscribe('iouThreshold', () => {
      this._triggerInference();
    });

    // Active image change
    state.subscribe('activeImage', (imgData) => {
      if (imgData && imgData.element) {
        this.canvasEngine.setImage(imgData.element);
        const emptyState = document.getElementById('viewportEmptyState');
        if (emptyState) emptyState.style.display = 'none';
      }
    });

    // Filter change
    state.subscribe('activeFilter', (filterClass) => {
      this.canvasEngine.setFilter(filterClass);
      document.querySelectorAll('.damage-item').forEach((item) => {
        const cls = item.getAttribute('data-class');
        item.style.borderColor = cls === filterClass ? 'var(--color-primary)' : '';
      });
    });

    // Detection update
    state.subscribe('detections', (dets) => {
      if (this.showDetections) {
        this.canvasEngine.setDetections(dets);
      }
      this._updateTelemetry(dets);
    });

    // Loading indicator
    state.subscribe('isDetecting', (isDetecting) => {
      const overlay = document.getElementById('stageLoadingOverlay');
      overlay?.classList.toggle('active', isDetecting);
    });
  }

  _updateTelemetry(dets) {
    const metrics = state.get('metrics');

    const valDetectionsCount = document.getElementById('valDetectionsCount');
    const valTotalLatency = document.getElementById('valTotalLatency');
    const valInferenceTime = document.getElementById('valInferenceTime');
    const valFps = document.getElementById('valFps');

    if (valDetectionsCount) valDetectionsCount.textContent = dets.length;
    if (valTotalLatency) valTotalLatency.textContent = metrics.latency_ms;
    if (valInferenceTime) {
      const infTime = metrics.speed.inference_ms || metrics.latency_ms;
      valInferenceTime.textContent = `${infTime} ms`;
    }
    if (valFps) {
      const fps = metrics.latency_ms > 0 ? (1000 / metrics.latency_ms).toFixed(1) : '0';
      valFps.textContent = fps;
    }

    // Damage counts
    const counts = { D00: 0, D10: 0, D20: 0, D40: 0 };
    dets.forEach((d) => {
      if (counts[d.class_id] !== undefined) {
        counts[d.class_id]++;
      }
    });

    const countD00 = document.getElementById('countD00');
    const countD10 = document.getElementById('countD10');
    const countD20 = document.getElementById('countD20');
    const countD40 = document.getElementById('countD40');

    if (countD00) countD00.textContent = counts.D00;
    if (countD10) countD10.textContent = counts.D10;
    if (countD20) countD20.textContent = counts.D20;
    if (countD40) countD40.textContent = counts.D40;

    // Raw boxes list
    const detsListEl = document.getElementById('rawDetectionsList');
    if (detsListEl) {
      if (dets.length === 0) {
        detsListEl.innerHTML = `<span style="color: var(--color-on-dark-soft); font-size: 10px;">No damages detected at current confidence.</span>`;
      } else {
        detsListEl.innerHTML = dets
          .map((d) => {
            const pct = Math.round(d.confidence * 100);
            return `
            <div class="detection-row">
              <span>${d.class_id} · ${d.class_name}</span>
              <span class="detection-confidence">${pct}%</span>
            </div>
          `;
          })
          .join('');
      }
    }
  }

  _updateStageMeta(title, dims) {
    const metaTitle = document.getElementById('stageImageTitle');
    const metaDims = document.getElementById('stageImageDims');
    if (metaTitle) metaTitle.textContent = title;
    if (metaDims) metaDims.textContent = dims;
  }

  _getModelDisplayName(modelId) {
    const names = {
      yolo11s_best: 'YOLO11s',
      yolov10s_best: 'YOLOv10s',
      yolov26n_best: 'YOLO26n',
      yolov8s_best: 'YOLOv8s',
    };
    return names[modelId] || modelId;
  }

  _showToast(message) {
    const toast = document.getElementById('floatingToast');
    const toastMsg = document.getElementById('floatingToastMsg');
    if (!toast || !toastMsg) return;

    toastMsg.textContent = message;
    toast.classList.add('active');

    if (this._toastTimer) clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => {
      toast.classList.remove('active');
    }, 2800);
  }
}

// Bootstrap on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new App();
  app.init();
});
