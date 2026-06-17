---
title: TrafficAI Bengaluru
emoji: 🚦
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# TrafficAI — Bengaluru Violation Detection System

AI-powered traffic enforcement system for safer roads. This application employs multiple YOLOv8 object detection models and EasyOCR in a parallelized pipeline to identify traffic violations in Bengaluru.

## Project Structure
```
traffic-violation-detector/
├── app.py                      # Main UI entrypoint & orchestration
├── config.py                   # Configuration and style tokens
├── requirements.txt            # Dependency listings
├── README.md                   # Space Configuration & documentation
├── utils/
│   ├── __init__.py
│   ├── model_loader.py        # Cached model loader helpers
│   ├── inference.py           # Parallel model inference logic
│   ├── ocr_processor.py       # Plate deduplication & OCR reading
│   ├── annotator.py           # Bounding box drawing & text styling
│   ├── report_generator.py    # Report compiling & downloads
│   └── visualizer.py          # Plotly dashboard generators
└── models/                     # Weights directory (user supplied)
    ├── helmet.pt
    ├── plate.pt
    ├── triple.pt
    ├── seatbelt.pt
    └── bengaluru.pt
```

## Running Locally

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the pre-trained YOLOv8 weights (`.pt` files) into the `models/` directory.
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. If weights are not available immediately, toggle **Simulation Mode** in the sidebar to test the entire application workflow.
