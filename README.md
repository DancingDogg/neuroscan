# NeuroScan — Ischemic Stroke Detection with Ensemble Deep Learning

A Final Year Project developed at **Universiti Tunku Abdul Rahman (UTAR)** under the Bachelor of Information Technology (Honours) Communications and Networking programme.

## Overview

NeuroScan is a Flask + Firebase web application that detects ischemic stroke from brain MRI scans using an ensemble of six deep learning models, with explainable AI visualisations for transparency.

## Features

- **6 Deep Learning Models** — ResNet50, ResNet101, DenseNet121, DenseNet169, EfficientNet-B3, Vision Transformer (ViT)
- **Ensemble Prediction** — Simple averaging, weighted averaging, hard voting, stacking
- **Explainable AI** — Grad-CAM (CNNs) and Attention Rollout (ViT)
- **Role-Based Access** — Patient, Doctor, Admin
- **Doctor Review Workflow** — Doctors can agree or disagree with AI predictions
- **Google OAuth** — Sign in / sign up with Google
- **Audit Logs** — All system actions are logged

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask, Python |
| Frontend | Jinja2, Vanilla JS, DM Sans |
| Database | Firebase Firestore |
| Auth | Firebase Authentication (Email + Google) |
| ML | PyTorch, timm, HuggingFace Transformers |
| XAI | Grad-CAM, Attention Rollout |

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/neuroscan.git
cd neuroscan
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your Firebase credentials
```

### 5. Add Firebase service account key
- Download `firebase-key.json` from Firebase Console → Project Settings → Service Accounts
- Place it in the project root (never commit this file)

### 6. Add model weights
- Place the `.pth` model files in `app/ml/model_files/`
- Model files are not included in this repository due to size constraints

### 7. Run the application
```bash
python run.py
```

Visit `http://127.0.0.1:5000`

## Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| ResNet50 | 1.000 | 1.000 | 1.000 |
| ResNet101 | 0.9706 | 0.9706 | 0.9827 |
| DenseNet121 | 0.9118 | 0.9118 | 0.9965 |
| DenseNet169 | 0.9412 | 0.9410 | 0.9970 |
| EfficientNet-B3 | 0.9706 | 0.9706 | 1.000 |
| Vision Transformer | 0.8824 | 0.8807 | 1.000 |
| **Ensemble (Best)** | **1.000** | **1.000** | **1.000** |

*Evaluated on a held-out test set of 34 MRI scans from Hospital Pengajar UPM (HPUPM).*

## Model Weights

The trained model weights are not included in this repository due to file size constraints.

Download from Google Drive: [Model Weights (Google Drive)](https://drive.google.com/drive/folders/1xQIKvAHJmykPyx3CiTvG2_-CQEpx4_UW?usp=drive_link)

After downloading, place all `.pth` files into `app/ml/model_files/`

## Project Structure

```
Brain Stroke Detection/
├── app/
│   ├── ml/                  # ML models and XAI
│   │   ├── model_files/     # .pth weights (not in repo)
│   │   ├── model_loader.py
│   │   ├── gradcam.py
│   │   └── vit_rollout.py
│   ├── static/              # CSS, JS
│   ├── templates/           # Jinja2 HTML templates
│   ├── routes.py
│   ├── models.py
│   └── __init__.py
├── .env.example
├── requirements.txt
└── run.py
```

## Developer

**Lee Ding Kuan**
UTAR, Faculty of ICT (Kampar) · 2025

Supervisor: Puan Nur Lyana Shahfiqa Bt Albashah