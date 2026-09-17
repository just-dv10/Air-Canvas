# 🎨 Air Canvas AI - High-Precision Hand Drawing & Smart Geometric Shapes

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/)

> **Air Canvas AI** is a real-time computer vision application that turns your webcam into a virtual drawing canvas. Draw, sketch, auto-convert rough sketches into clean geometric shapes, connect multi-shape diagrams, and interact with mid-air hand gestures—no physical touch, pens, or sensors required!

---

## ✨ Key Features

- 🎯 **Rock-Solid Fingertip Tracking**: Velocity-adaptive Exponential Smoothing (EMA) filter completely eliminates webcam noise and hand tremors for zero-jitter, razor-sharp lines.
- 🤏 **Pinch-to-Draw Precision**: Touch your Index finger and Thumb together to draw; release to stop. Prevents accidental marks when moving your hands.
- 📐 **Smart Geometric Shape Auto-Detection**:
  - Draw rough strokes in mid-air—Air Canvas instantly recognizes and fits clean vector shapes:
    - **Straight Lines** (auto-straightened)
    - **Rectangles & Boxes** (crisp 90-degree orthogonal corners)
    - **Circles & Ellipses** (smooth circular contours)
    - **Triangles** (3-vertex polygon fitting)
    - **Freehand Curves** (smooth polyline interpolation)
- 🧲 **Magnetic Anchor Snapping (Connect Shapes)**:
  - Every shape generates connectable anchor points at its corners, endpoints, and centers.
  - Starting or ending a stroke near an existing anchor automatically snaps to it, allowing you to easily build connected shapes, multi-shape diagrams, house blueprints, polygons, and flowcharts.
- 🖐️ **Open-Palm Clear with Radial Countdown**:
  - Hold an open palm (all 5 fingers spread) for 1 second to trigger an on-screen radial wipe meter and safely clear your screen.
- ↩️ **Undo Support**: Tap the Undo button on the toolbar or press `Z` to roll back shapes.
- 🎨 **Interactive Floating Toolbar**:
  - Palette: **Cyan**, **Neon Pink**, **Emerald Green**, **Amber**, **White**, **Eraser**, **Undo**, and **Clear**.
- 🖼️ **Dual View Modes**:
  - **AR Blend View**: Drawings overlaid directly on your live video stream.
  - **Blackboard View**: Solid dark digital canvas for clean, distraction-free artwork.
- 💾 **Snapshot Export**: Press `S` to save your artwork to `saved_drawings/` with a timestamp.

---

## 🎮 Gesture Cheat Sheet

| Gesture | Fingers / Hand | Action |
| :--- | :---: | :--- |
| 🤏 **Index + Thumb Pinch** | Pinch (Touch together) | **Draw Mode**: Paints live stroke on canvas or taps toolbar buttons. |
| ✋ **Pinch Released** | Separate Fingers | **Auto-Shape**: Analyzes stroke and converts it into a clean shape. |
| 🧲 **Near Corner/Vertex** | Hover < 28px from anchor | **Magnetic Snap**: Snaps start or end point to connect shapes together. |
| 🖐️ **Open Palm (5 Fingers)** | Spread hand & hold ~1s | **Canvas Clear**: Activates circular progress wipe to clear canvas. |
| 👆 **Hover / Navigation** | No pinch | **Free Cursor**: Move crosshair across screen without drawing. |

---

## ⌨️ Keyboard Shortcuts

| Key | Description |
| :---: | :--- |
| `Z` | **Undo**: Removes the last committed shape and restores previous anchors. |
| `C` | **Clear Canvas**: Instantly erases all drawings and shapes. |
| `S` | **Save Artwork**: Exports current drawing as `.png` into `saved_drawings/`. |
| `B` | **Toggle View**: Switches between AR camera overlay and pure Blackboard mode. |
| `+` / `=` | **Increase Brush Size**: Enlarges drawing/eraser radius. |
| `-` / `_` | **Decrease Brush Size**: Shrinks drawing/eraser radius. |
| `Q` | **Quit**: Closes the application safely. |

---

## 📁 Project Structure

```text
air-canvas-ai/
├── .gitignore                  # Ignored files (venv, models, caches, snapshots)
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
├── requirements.txt            # Project dependencies
├── src/
│   ├── __init__.py
│   ├── stabilizer.py           # Adaptive velocity-aware jitter filter
│   ├── shape_recognizer.py     # Smart shape classification, fitting & anchor snapping
│   ├── hand_tracker.py         # MediaPipe tracking (Tasks & Solutions compatible)
│   ├── canvas.py               # Shape repository, anchor manager, live preview & blending
│   ├── ui_overlay.py           # Toolbar buttons, magnetic snap HUD, palm clear radial meter
│   └── main.py                 # Main application loop & camera capture
└── tests/
    ├── test_canvas.py          # Unit tests for canvas, strokes, undo, and toolbar
    ├── test_shape_recognizer.py# Unit tests for geometric shape fitting & snapping
    └── test_hand_tracker.py    # Unit tests for tracker and stabilizers
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.8 to 3.12 installed.
- A functional webcam or USB camera.

### 2. Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/air-canvas-ai.git
cd air-canvas-ai
```

### 3. Set Up Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

Start the application with default settings:
```bash
python -m src.main
```

### Custom Arguments
```bash
# Run with camera index 1 and 1920x1080 resolution
python -m src.main --camera 1 --width 1920 --height 1080
```

---

## 🧪 Running Automated Tests

Run the automated offline unit tests (runs without camera hardware):
```bash
python -m unittest discover -s tests
```

---

## 📤 How to Upload to GitHub

1. **Create a new repository on GitHub**:
   - Go to [github.com/new](https://github.com/new)
   - Repository name: `air-canvas-ai`
   - Set visibility to **Public** (or **Private**).
   - Leave "Add README" and ".gitignore" **unchecked**.
   - Click **Create repository**.

2. **Push your code from terminal**:
   ```bash
   cd air-canvas-ai

   # Stage any new changes
   git add .

   # Commit
   git commit -m "feat: add jitter stabilization, pinch drawing, and smart shape snapping"

   # Rename branch to main
   git branch -M main

   # Add your remote repository (replace with your username)
   git remote add origin https://github.com/<YOUR-USERNAME>/air-canvas-ai.git

   # Push to GitHub
   git push -u origin main
   ```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
