/**
 * AIL-RDD Canvas Engine
 * High-DPI canvas rendering, bounding box overlays, damage pills, and pan/zoom.
 * Strictly zero emojis.
 */

const CLASS_COLORS = {
  D00: { border: '#5db872', fill: 'rgba(93, 184, 114, 0.14)', text: '#141413' },
  D10: { border: '#5db8a6', fill: 'rgba(93, 184, 166, 0.14)', text: '#141413' },
  D20: { border: '#cc785c', fill: 'rgba(204, 120, 92, 0.16)', text: '#ffffff' },
  D40: { border: '#e8a55a', fill: 'rgba(232, 165, 90, 0.16)', text: '#141413' },
};

const DEFAULT_COLOR = {
  border: '#cc785c',
  fill: 'rgba(204, 120, 92, 0.15)',
  text: '#ffffff',
};

export class DetectionCanvas {
  constructor(canvasElement, containerElement) {
    this.canvas = canvasElement;
    this.container = containerElement;
    this.ctx = this.canvas.getContext('2d');

    this.image = null;
    this.detections = [];
    this.filterClass = 'all';

    // Transform State
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.startX = 0;
    this.startY = 0;

    this._bindEvents();
  }

  setImage(imageElement) {
    this.image = imageElement;
    this.resetTransform();
    this.render();
  }

  setDetections(detections) {
    this.detections = detections || [];
    this.render();
  }

  setFilter(className) {
    this.filterClass = className || 'all';
    this.render();
  }

  resetTransform() {
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.render();
  }

  resize() {
    if (!this.container) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.container.getBoundingClientRect();
    const width = Math.floor(rect.width);
    const height = Math.floor(rect.height || 540);

    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;

    this.ctx.scale(dpr, dpr);
    this.render();
  }

  render() {
    if (!this.container) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.container.getBoundingClientRect();
    const cw = rect.width;
    const ch = rect.height || 540;

    // Clear
    this.ctx.save();
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.clearRect(0, 0, cw, ch);

    if (!this.image) {
      this.ctx.restore();
      return;
    }

    // Apply pan & zoom
    this.ctx.translate(cw / 2 + this.panX, ch / 2 + this.panY);
    this.ctx.scale(this.zoom, this.zoom);

    // Calculate image fitting aspect ratio
    const imgW = this.image.naturalWidth || this.image.width;
    const imgH = this.image.naturalHeight || this.image.height;
    const scale = Math.min((cw * 0.92) / imgW, (ch * 0.92) / imgH);
    const dw = imgW * scale;
    const dh = imgH * scale;
    const dx = -dw / 2;
    const dy = -dh / 2;

    // Draw Image
    this.ctx.drawImage(this.image, dx, dy, dw, dh);

    // Draw Detections
    const coordScaleX = dw / imgW;
    const coordScaleY = dh / imgH;

    for (const det of this.detections) {
      if (this.filterClass !== 'all' && det.class_id !== this.filterClass) {
        continue;
      }

      const [x1, y1, x2, y2] = det.bbox;
      const bx = dx + x1 * coordScaleX;
      const by = dy + y1 * coordScaleY;
      const bw = (x2 - x1) * coordScaleX;
      const bh = (y2 - y1) * coordScaleY;

      const palette = CLASS_COLORS[det.class_id] || DEFAULT_COLOR;

      // Fill Box
      this.ctx.fillStyle = palette.fill;
      this.ctx.fillRect(bx, by, bw, bh);

      // Stroke Box
      this.ctx.strokeStyle = palette.border;
      this.ctx.lineWidth = Math.max(1.5, 2 / this.zoom);
      this.ctx.strokeRect(bx, by, bw, bh);

      // Draw Badge Pill
      const confPercent = Math.round(det.confidence * 100);
      const labelText = `${det.class_id} · ${confPercent}%`;
      this.ctx.font = '500 11px "JetBrains Mono", monospace';

      const textWidth = this.ctx.measureText(labelText).width;
      const pillW = textWidth + 12;
      const pillH = 18;
      const pillX = bx;
      const pillY = Math.max(dy, by - pillH - 4);

      // Pill Background
      this.ctx.fillStyle = palette.border;
      this._roundRect(pillX, pillY, pillW, pillH, 4);
      this.ctx.fill();

      // Pill Text
      this.ctx.fillStyle = palette.text;
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(labelText, pillX + 6, pillY + pillH / 2);
    }

    this.ctx.restore();
  }

  _roundRect(x, y, w, h, r) {
    this.ctx.beginPath();
    this.ctx.moveTo(x + r, y);
    this.ctx.lineTo(x + w - r, y);
    this.ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    this.ctx.lineTo(x + w, y + h - r);
    this.ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    this.ctx.lineTo(x + r, y + h);
    this.ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    this.ctx.lineTo(x, y + r);
    this.ctx.quadraticCurveTo(x, y, x + r, y);
    this.ctx.closePath();
  }

  _bindEvents() {
    window.addEventListener('resize', () => this.resize());

    // Wheel Zoom
    this.container.addEventListener(
      'wheel',
      (e) => {
        if (!this.image) return;
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.12 : 0.89;
        this.zoom = Math.min(Math.max(0.5, this.zoom * factor), 5.0);
        this.render();
      },
      { passive: false }
    );

    // Pan / Drag
    this.container.addEventListener('mousedown', (e) => {
      if (!this.image) return;
      this.isDragging = true;
      this.startX = e.clientX - this.panX;
      this.startY = e.clientY - this.panY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      this.panX = e.clientX - this.startX;
      this.panY = e.clientY - this.startY;
      this.render();
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    // Touch Support
    let lastTouchDist = 0;
    this.container.addEventListener(
      'touchstart',
      (e) => {
        if (!this.image) return;
        e.preventDefault();
        if (e.touches.length === 1) {
          this.isDragging = true;
          this.startX = e.touches[0].clientX - this.panX;
          this.startY = e.touches[0].clientY - this.panY;
        } else if (e.touches.length === 2) {
          lastTouchDist = Math.hypot(
            e.touches[0].clientX - e.touches[1].clientX,
            e.touches[0].clientY - e.touches[1].clientY
          );
        }
      },
      { passive: false }
    );

    this.container.addEventListener(
      'touchmove',
      (e) => {
        if (!this.image) return;
        e.preventDefault();
        if (e.touches.length === 1 && this.isDragging) {
          this.panX = e.touches[0].clientX - this.startX;
          this.panY = e.touches[0].clientY - this.startY;
          this.render();
        } else if (e.touches.length === 2) {
          const dist = Math.hypot(
            e.touches[0].clientX - e.touches[1].clientX,
            e.touches[0].clientY - e.touches[1].clientY
          );
          if (lastTouchDist > 0) {
            const factor = dist / lastTouchDist;
            this.zoom = Math.min(Math.max(0.5, this.zoom * factor), 5.0);
            this.render();
          }
          lastTouchDist = dist;
        }
      },
      { passive: false }
    );

    this.container.addEventListener('touchend', () => {
      this.isDragging = false;
      lastTouchDist = 0;
    });
  }
}
