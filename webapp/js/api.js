/**
 * AIL-RDD API Client & Resilient Data Adapter
 * Supports fullstack PyTorch backend and graceful client-side fallback.
 * Strictly zero emojis.
 */

class ApiClient {
  constructor() {
    this.baseUrl = window.location.origin;
    this.isLive = false;
    this._cachedSamples = null;
    this._cachedBenchmark = null;
  }

  async checkStatus() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1800);
      const res = await fetch(`${this.baseUrl}/api/models`, {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        const data = await res.json();
        this.isLive = data.status === 'online';
        return { isLive: this.isLive, data };
      }
    } catch {
      this.isLive = false;
    }
    return { isLive: false, data: null };
  }

  async fetchModels() {
    if (this.isLive) {
      try {
        const res = await fetch(`${this.baseUrl}/api/models`);
        if (res.ok) {
          const data = await res.json();
          return data.models || [];
        }
      } catch (err) {
        console.warn('Falling back to static benchmark data:', err);
      }
    }

    // Static fallback
    if (this._cachedBenchmark) return this._cachedBenchmark;
    try {
      const res = await fetch('data/benchmark.json');
      if (res.ok) {
        this._cachedBenchmark = await res.json();
        return this._cachedBenchmark;
      }
    } catch (err) {
      console.error('Failed to load static benchmark.json:', err);
    }
    return [];
  }

  async fetchSamples() {
    if (this.isLive) {
      try {
        const res = await fetch(`${this.baseUrl}/api/samples`);
        if (res.ok) {
          const data = await res.json();
          this._cachedSamples = data;
          return data;
        }
      } catch (err) {
        console.warn('Falling back to static samples data:', err);
      }
    }

    // Static fallback
    if (this._cachedSamples) return this._cachedSamples;
    try {
      const res = await fetch('data/samples.json');
      if (res.ok) {
        this._cachedSamples = await res.json();
        return this._cachedSamples;
      }
    } catch (err) {
      console.error('Failed to load static samples.json:', err);
    }
    return [];
  }

  async fetchDatasetImages(offset = 0, limit = 10) {
    if (this.isLive) {
      try {
        const res = await fetch(`${this.baseUrl}/api/dataset-images?offset=${offset}&limit=${limit}`);
        if (res.ok) {
          return await res.json();
        }
      } catch (err) {
        console.warn('Failed to fetch dataset images from backend:', err);
      }
    }

    // Static fallback from dataset_catalog.json
    try {
      const res = await fetch('data/dataset_catalog.json');
      if (res.ok) {
        const data = await res.json();
        const allImgs = data.images || [];
        const batch = allImgs.slice(offset, offset + limit);
        return {
          total: data.total || allImgs.length,
          offset,
          limit,
          has_more: offset + limit < (data.total || allImgs.length),
          images: batch,
        };
      }
    } catch (err) {
      console.error('Failed to load dataset_catalog.json fallback:', err);
    }

    // Default to samples if dataset_catalog fails
    const samples = await this.fetchSamples();
    const batch = samples.slice(offset, offset + limit).map((s) => ({
      id: s.id,
      filename: s.file.replace('samples/', ''),
      url: s.file,
      title: s.title,
    }));
    return {
      total: samples.length,
      offset,
      limit,
      has_more: offset + limit < samples.length,
      images: batch,
    };
  }

  async detect({ model, conf, iou, sampleId, imageBase64 }) {
    if (this.isLive) {
      if (this.activeAbortController) {
        this.activeAbortController.abort();
      }
      this.activeAbortController = new AbortController();

      try {
        const payload = {
          model,
          conf: parseFloat(conf),
          iou: parseFloat(iou),
        };

        if (sampleId) {
          payload.sample_id = sampleId;
        }
        if (imageBase64) {
          payload.image_base64 = imageBase64;
        }

        const res = await fetch(`${this.baseUrl}/api/detect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: this.activeAbortController.signal,
        });

        if (res.ok) {
          return await res.json();
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          return null; // Gracefully ignore superseded requests
        }
        console.warn('Live detect failed, checking fallback:', err);
      }
    }

    // Standalone fallback using precomputed detections from samples
    const samples = await this.fetchSamples();
    const foundSample = samples.find((s) => s.id === sampleId);

    if (foundSample && foundSample.predictions) {
      const modelPreds = foundSample.predictions[model] || {};
      const allDets = modelPreds.detections || [];
      const filtered = allDets.filter((d) => d.confidence >= conf);

      return {
        model,
        device: 'client-cache',
        width: foundSample.width,
        height: foundSample.height,
        count: filtered.length,
        detections: filtered,
        latency_ms: modelPreds.latency_ms || 28.4,
        speed: {
          preprocess_ms: 1.1,
          inference_ms: modelPreds.latency_ms || 28.4,
          postprocess_ms: 0.9,
        },
      };
    }

    // For uploaded custom images without backend
    return {
      model,
      device: 'offline-notice',
      count: 0,
      detections: [],
      latency_ms: 0,
      notice: 'Start python3 webapp/main.py for live custom image inference.',
      speed: { preprocess_ms: 0, inference_ms: 0, postprocess_ms: 0 },
    };
  }
}

export const api = new ApiClient();
