# Textile Surface Defect Detection Using Computer Vision

Automated detection of fabric surface defects (stains, holes, scratches, irregular
patterns) using OpenCV with an optional YOLO/CNN backend, delivered through a
Streamlit web interface.

## Features
- Upload fabric images or process a whole folder in batch.
- Defect highlighting with bounding boxes and contours.
- Defect mask and deviation map visualisation.
- Severity grading and CSV report export.
- Optional YOLO backend (when `ultralytics` is installed and weights provided).
- Synthetic sample images for an immediate demo.

## Tech stack
- Python 3.x
- OpenCV (computer vision)
- NumPy / Pandas / Matplotlib
- Streamlit (web UI)
- Optional: Ultralytics YOLO (deep learning)

## Setup
```bash
pip install -r requirements.txt
python generate_samples.py
streamlit run app.py
```

## Dataset
Fabric surface defect dataset (Mendeley):
https://data.mendeley.com/datasets/6574nhzm8x/1

Reference paper:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11648198/

## MATLAB version
A MATLAB implementation is available in `matlab/`. See `matlab/README_MATLAB.md`.

```matlab
cd('matlab')
generateSamples('../samples')
runDemo('../samples/sample_1.png')
runBatch('../samples', '../outputs')
```
```bash
pip install ultralytics
python train_yolo.py --data path/to/data.yaml --epochs 50
```
See `train_yolo.py` for a ready-to-use training template.

## Project structure
```
.
├── app.py
├── generate_samples.py
├── train_yolo.py
├── evaluate.py
├── requirements.txt
├── matlab/
│   ├── detectDefects.m
│   ├── runDemo.m
│   ├── runBatch.m
│   ├── generateSamples.m
│   └── README_MATLAB.md
├── src/
│   ├── detector.py
│   └── utils.py
├── samples/
├── outputs/
└── models/
```
