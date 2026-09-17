# 🎨 Air Canvas AI - Paint in the Air with Hand Gestures

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/)

> **Air Canvas AI** is a real-time computer vision application that turns your webcam into a virtual canvas. Draw, sketch, and interact with on-screen tools using natural hand gestures in mid-air—no physical touch, pens, or sensors required!

---

## ✨ Features

- 🖐️ **Real-Time Hand Landmark Tracking**: High-accuracy fingertip tracking powered by Google's MediaPipe Hands.
- 🖌️ **Natural Gesture Controls**:
  - **Drawing Mode**: Raise only your index finger to paint smooth, anti-aliased strokes.
  - **Selection / Hover Mode**: Raise both index and middle fingers to navigate, hover, and select colors or tools.
  - **Standby Mode**: Relax your hand or make a fist to pause drawing without leaving the screen.
- 🎨 **Interactive Floating Toolbar**:
  - Color palette: **Cyan**, **Neon Pink**, **Emerald Green**, **Amber**, and **White**.
  - Integrated **Eraser** tool with wide radius.
  - One-tap **Clear** button.
- 🖼️ **Dual View Modes**:
  - **AR Blend Mode**: Your drawings are overlaid directly onto your live camera stream.
  - **Blackboard Mode**: Switch to a solid dark canvas for clean, distraction-free digital artwork.
- 💾 **Snapshot Export**: Press `S` anytime to save your artwork to the `saved_drawings/` folder with timestamp.
- 📏 **Dynamic Brush Resizing**: Adjust brush and eraser stroke width on the fly.

---

## 🎮 Gesture Cheat Sheet

| Gesture | Fingers Up | Mode | Action |
| :--- | :---: | :---: | :--- |
| ☝️ **Index Up Only** | Index (1) | **Drawing Mode** | Paints on the canvas at your fingertip location. |
| ✌️ **Index + Middle Up** | Index + Middle (2) | **Selection Mode** | Free cursor. Hover over top buttons to pick colors or tools. |
| ✋ **Open Palm / Fist** | None or All (3+) | **Standby Mode** | Idle state; prevents accidental marks while talking or moving. |

---

## ⌨️ Keyboard Shortcuts

| Key | Description |
| :---: | :--- |
| `C` | **Clear Canvas**: Instantly erases all drawings. |
| `S` | **Save Artwork**: Exports current drawing as `.png` into `saved_drawings/`. |
| `B` | **Toggle View**: Switches between AR camera overlay and pure Blackboard mode. |
| `+` / `=` | **Increase Brush Size**: Enlarges drawing/eraser radius. |
| `-` / `_` | **Decrease Brush Size**: Shrinks drawing/eraser radius. |
| `Q` | **Quit**: Closes the application safely. |

---

## 📁 Project Structure

```text
air-canvas-ai/
├── .gitignore               # Ignored files (venv, screenshots, caches)
├── LICENSE                  # MIT License
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
├── src/
│   ├── __init__.py
│   ├── hand_tracker.py      # MediaPipe hand detection and finger tracking wrapper
│   ├── canvas.py            # Drawing layer, stroke interpolation, and blending
│   ├── ui_overlay.py        # Top toolbar, buttons, and HUD feedback
│   └── main.py              # Main execution loop and camera capture
└── tests/
    └── test_canvas.py       # Offline unit tests for canvas and UI logic
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.8 to 3.12 installed.
- A functional webcam or external USB camera.

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
If you have multiple webcams or wish to run at a custom resolution:
```bash
# Run with camera index 1 and 1920x1080 resolution
python -m src.main --camera 1 --width 1920 --height 1080
```

---

## 🧪 Running Tests

Run the automated offline unit tests (does not require camera access):
```bash
python -m unittest discover -s tests
```

---

## 📤 How to Upload to GitHub

Follow these steps to publish this project to your GitHub account:

1. **Create a new repository on GitHub**:
   - Go to [github.com/new](https://github.com/new)
   - Repository name: `air-canvas-ai`
   - Description: *Computer vision-based Air Canvas using Python, OpenCV, and MediaPipe.*
   - Set repository to **Public** (or **Private**).
   - **Do NOT** check "Add a README file" or ".gitignore" (we already have them).
   - Click **Create repository**.

2. **Initialize Git & Push from your terminal**:
   ```bash
   cd air-canvas-ai

   # Initialize git repository
   git init

   # Stage all files
   git add .

   # Create initial commit
   git commit -m "feat: initial commit of Air Canvas AI application"

   # Rename branch to main
   git branch -M main

   # Link your remote GitHub repository
   git remote add origin https://github.com/<YOUR-USERNAME>/air-canvas-ai.git

   # Push your code
   git push -u origin main
   ```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
