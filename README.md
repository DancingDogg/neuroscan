---
title: NeuroScan
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

<h1 align="center">🧠 NeuroScan</h1>
<p align="center">Ischemic Stroke Detection with Ensemble Deep Learning</p>
<p align="center">
  <a href="https://dancinggdogg-neuroscan.hf.space"><strong>🚀 Live Demo</strong></a> ·
  <a href="https://github.com/DancingDogg/neuroscan"><strong>GitHub</strong></a>
</p>

---

A Final Year Project developed at **Universiti Tunku Abdul Rahman (UTAR)** under the Bachelor of Information Technology (Honours) Communications and Networking programme.

## Overview

NeuroScan is a Flask + Firebase web application that detects ischemic stroke from brain MRI scans using an ensemble of six deep learning models, with explainable AI visualisations for transparency.

## Features

- **6 Deep Learning Models** — ResNet50, ResNet101, DenseNet121, DenseNet169, EfficientNet-B3, Vision Transformer (ViT)
- **Ensemble Prediction** — Simple averaging, weighted averaging, hard voting, stacking
- **Explainable AI** — Grad-CAM (CNNs) and Attention Rollout (ViT)
- **Role-Based Access** — Patient, Doctor, Admin
- **Doctor Review Workflow** — Doctors can agree or disagree with AI predictions
- **AI Chatbot** — ILMU, powered by Claude Haiku, for stroke education
- **Google OAuth** — Sign in / sign up with Google
- **PDF Export** — Patients can export prediction history as a PDF report
- **Security** — Flask-Talisman CSP, rate limiting, magic byte validation, audit logging

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask, Python |
| Frontend | Jinja2, Vanilla JS |
| Database | Firebase Firestore |
| Auth | Firebase Authentication (Email + Google OAuth) |
| ML | PyTorch, timm, HuggingFace Transformers |
| XAI | Grad-CAM, Attention Rollout |
| AI Chatbot | Anthropic Claude Haiku |
| Security | Flask-Talisman, Flask-Limiter, Flask-Login |
| Deployment | Docker, Hugging Face Spaces |

## Live Demo

👉 **https://dancinggdogg-neuroscan.hf.space**

> Note: First load may take 5–10 minutes as model weights are downloaded on cold start.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| ResNet50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ResNet101 | 0.9706 | 0.9722 | 0.9706 | 0.9706 | 0.9827 |
| DenseNet121 | 0.9118 | 0.9167 | 0.9118 | 0.9118 | 0.9965 |
| DenseNet169 | 0.9412 | 0.9444 | 0.9412 | 0.9410 | 0.9970 |
| EfficientNet-B3 | 0.9706 | 0.9722 | 0.9706 | 0.9706 | 1.000 |
| Vision Transformer | 0.8824 | 0.8873 | 0.8824 | 0.8807 | 1.000 |
| **Ensemble (Best)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

*Evaluated on a held-out test set of 34 MRI scans from Hospital Pengajar UPM (HPUPM).*

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/DancingDogg/neuroscan.git
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
# Edit .env with your Firebase and API credentials
```

### 5. Add Firebase service account key
- Download `firebase-key.json` from Firebase Console → Project Settings → Service Accounts
- Place it in the project root (never commit this file)

### 6. Add model weights
- Place all `.pth` files in `app/ml/model_files/`
- Download from: [Google Drive](https://drive.google.com/drive/folders/1xQIKvAHJmykPyx3CiTvG2_-CQEpx4_UW?usp=drive_link)

### 7. Run the application
```bash
python run.py
```

Visit `http://127.0.0.1:5000`

## Project Structure

```
Brain Stroke Detection/
├── app/
│   ├── ml/                  # ML models and XAI
│   │   ├── model_files/     # .pth weights (not in repo)
│   │   ├── model_loader.py
│   │   ├── gradcam.py
│   │   └── vit_rollout.py
│   ├── static/              # CSS, JS, favicon
│   ├── templates/           # Jinja2 HTML templates
│   ├── routes.py
│   ├── models.py
│   └── __init__.py
├── Dockerfile
├── .env.example
├── requirements.txt
└── run.py
```

## Known Limitations

- Email notifications require unrestricted outbound SMTP — blocked on Hugging Face free tier
- GradCAM images are ephemeral on HF free tier (lost on container restart)
- Cold start takes 5–10 minutes to download model weights (~697 MB)
- Predictions run on CPU only on free tier — slower than local GPU inference

## Developer

**Lee Ding Kuan**
UTAR, Faculty of Information and Communication Technology (Kampar) · 2025
Supervisor: Puan Nur Lyana Shahfiqa Bt Albashah