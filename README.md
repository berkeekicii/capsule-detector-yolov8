# Capsule Defect Detection with YOLOv8

An AI-based quality inspection system for coffee capsules. It uses YOLOv8 object detection to find **damaged** and **intact** capsules in an image, and ships with a small Streamlit web app for trying it out.

## Overview

The goal of the project is to automate a visual quality check: given a photo of coffee capsules, the model locates each capsule and labels it as intact or damaged.

I built the project end to end: I collected and annotated a custom image dataset, trained and evaluated object detection models on both RGB and grayscale versions of the images, and wrapped the trained detector in a simple web interface.

**Classes**

| ID | Class |
| --- | --- |
| 0 | `damaged_capsule` |
| 1 | `intact_capsule` |

## Dataset

The dataset was annotated and exported with Roboflow in YOLOv8 format and is stored in `dataset/` (`train`, `valid`, `test` splits plus `data.yaml`).

- 717 images, 2 classes
- Pre-processing: auto-orientation and resize to 640x640 (fit with black edges)
- No image augmentation was applied in the exported version
- Roboflow project: [coffee-capsule-defect-detection](https://universe.roboflow.com/8250898-estg-ipp-pt/coffee-capsule-defect-detection/dataset/1), licensed CC BY 4.0

## Repository structure

```text
.
├── app/
│   ├── webapp.py      # Streamlit web app
│   └── app.py         # command-line / OpenCV window demo for one image
├── dataset/           # YOLOv8-format dataset exported from Roboflow
├── notebook/          # YOLOv8 notebook
├── test/              # sample test image
└── requirements.txt
```

## Getting started

Install the dependencies (a virtual environment is recommended):

```bash
pip install -r requirements.txt
```

The trained weights are not stored in this repository. Train the model with the notebook and the dataset, then place the result at `weights/best.pt`, or point to it with an environment variable:

```bash
# Linux / macOS
export CAPSULE_MODEL=path/to/best.pt

# Windows PowerShell
$env:CAPSULE_MODEL = "path\to\best.pt"
```

Start the web app:

```bash
streamlit run app/webapp.py
```

Upload a `.jpg`, `.jpeg` or `.png` image of coffee capsules. The app resizes it to 640x640 without enlarging it (letterbox with black padding), runs the detector with a confidence threshold of 0.5 and shows the image with the predicted boxes.

To test a single image from the command line instead:

```bash
python app/app.py test/foto1.jpg weights/best.pt
```

## Technologies

Python, YOLOv8 (Ultralytics), PyTorch, CUDA, OpenCV, Streamlit, Roboflow.

## Notes

- Letterboxing never zooms in. Images are only shrunk and padded, which keeps small defects at their original scale.
- This is a student project built on a small custom dataset. Results will depend on lighting, background and camera, so it should be treated as a prototype and not as a production inspection system.
