# AIL-RDD Road Damage Detection Web Application

A high-performance browser interface for road distress detection and architectural benchmark analysis, designed according to the warm-canvas editorial design system in `DESIGN.md`.

## Features

- **Multi-Model Dynamic Switcher**: Seamless switching between 4 trained architectures:
  - **YOLO11s**: Highest precision (81.49% mAP@50)
  - **YOLOv10s**: NMS-free end-to-end architecture (77.26% mAP@50)
  - **YOLO26n (DGVG)**: Ultra-lightweight mobile architecture with custom Deformable Convolutions & VoVGSCSP (1.89M params, 3.8 MB footprint)
  - **YOLOv8s**: Production baseline (79.79% mAP@50)
- **Floating Design**: Floating frosted glass navigation, floating bottom control dock, and elevated card surfaces.
- **Scroll Fade Effects**: Smooth element entrance transitions on scroll and gradient edge masks on scrollable containers.
- **High-DPI Interactive Viewport**: Retina-ready canvas with bounding boxes, damage classification tags, confidence ratings, and pan/zoom interaction.
- **Dual Execution Modes**:
  - **Live Fullstack Engine**: Real-time PyTorch/Ultralytics GPU/MPS/CPU inference via `webapp/server.py`.
  - **Standalone Browser Mode**: Direct browser launch with curated samples and precomputed benchmark telemetry.
- **Zero-Emoji Discipline**: Strict adherence to geometric SVG iconography and editorial typography.

## Quick Start

### Mode 1: Live Engine (PyTorch & Ultralytics)

From the project root:

```bash
# Using Python virtual environment
.venv/bin/python webapp/server.py

# Or standard python3
python3 webapp/server.py
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
