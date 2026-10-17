# AIL-RDD Road Damage Detection Web Application

A high-performance fixed-viewport browser interface for road distress detection and architectural benchmark analysis, designed according to the warm-canvas editorial design system in `DESIGN.md`.

## Features

- **Fixed-Frame Viewport Layout**: Locked full-height application layout (`100vh`) eliminating cumbersome window scrolling, with dedicated internal scroll containers featuring smooth edge fade gradient masks.
- **Unified Command Sidebar**:
  - Replaces top and bottom bars with a streamlined left control deck.
  - Quick view switching: **Studio View**, **Benchmark Matrix**, **RDD Taxonomy**.
  - Direct 4-model architecture switcher (YOLO11s, YOLOv10s, YOLO26n DGVG, YOLOv8s).
  - Fine-grained Confidence and IoU threshold sliders.
  - Image upload and detection triggers.
  - Live hardware acceleration status badge (CPU/MPS/CUDA).
- **5-Column Vertical Dataset Gallery**:
  - Chunked pagination directly from the 3,543 test images of the RDD2022 dataset.
  - Loads 10 images initially (2 rows of 5).
  - Dedicated **Tải thêm (+10)** button to load subsequent batches of 10 images dynamically.
  - Vertical scrolling with top and bottom CSS gradient masks (`.scroll-mask-y`).
- **High-DPI Interactive Viewport**: Retina-ready canvas with bounding boxes, damage classification tags, confidence ratings, and pan/zoom interaction.
- **Multi-Model Dynamic Switcher**: Seamless switching between:
  - **YOLO11s**: Highest precision (81.49% mAP@50)
  - **YOLOv10s**: NMS-free end-to-end architecture (77.26% mAP@50)
  - **YOLO26n (DGVG)**: Ultra-lightweight mobile architecture with custom Deformable Convolutions & VoVGSCSP (1.89M params, 3.8 MB footprint)
  - **YOLOv8s**: Production baseline (79.79% mAP@50)
- **Zero-Emoji Discipline**: Strict adherence to geometric SVG iconography and editorial typography.

## Quick Start

### Mode 1: Live Engine (PyTorch & Ultralytics)

From the project root:

```bash
# Using Python virtual environment
.venv/bin/python webapp/main.py

# Or standard python3
python3 webapp/main.py
```

Then open your browser at:
```
http://localhost:8000
```

### Mode 2: Standalone Static Mode

Open `webapp/index.html` directly in any web browser, or use any static file server:

```bash
cd webapp
python3 -m http.server 8000
```
