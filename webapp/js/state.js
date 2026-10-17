/**
 * AIL-RDD Reactive State Management
 * Ultra-minimalist pub-sub state container. Zero external dependencies.
 */

class State {
  constructor() {
    this._state = {
      selectedModel: 'yolo11s_best',
      confidenceThreshold: 0.25,
      iouThreshold: 0.45,
      activeImage: null, // { url, width, height, id, file, element }
      detections: [],
      metrics: {
        latency_ms: 0,
        count: 0,
        speed: { preprocess_ms: 0, inference_ms: 0, postprocess_ms: 0 },
      },
      isDetecting: false,
      models: [],
      samples: [],
      isBackendLive: false,
      activeFilter: 'all',
      zoom: 1.0,
      pan: { x: 0, y: 0 },
    };

    this._listeners = new Map();
  }

  get(key) {
    return this._state[key];
  }

  set(key, value) {
    if (this._state[key] === value) return;
    const oldValue = this._state[key];
    this._state[key] = value;
    this._notify(key, value, oldValue);
  }

  update(key, fn) {
    const newVal = fn(this._state[key]);
    this.set(key, newVal);
  }

  subscribe(key, callback) {
    if (!this._listeners.has(key)) {
      this._listeners.set(key, new Set());
    }
    this._listeners.get(key).add(callback);
    return () => {
      this._listeners.get(key).delete(callback);
    };
  }

  _notify(key, value, oldValue) {
    if (this._listeners.has(key)) {
      this._listeners.get(key).forEach((cb) => {
        try {
          cb(value, oldValue);
        } catch (err) {
          console.error(`State listener error on "${key}":`, err);
        }
      });
    }
    if (this._listeners.has('*')) {
      this._listeners.get('*').forEach((cb) => {
        try {
          cb(key, value, oldValue);
        } catch (err) {
          console.error('State wildcard listener error:', err);
        }
      });
    }
  }
}

export const state = new State();
