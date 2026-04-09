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

NeuroScan is a Flask + Firebase web application for ischemic stroke detection from brain MRI scans. It uses an ensemble of six deep learning models with explainable AI visualisations, a three-tier role system (Patient, Doctor, Admin), and a built-in AI medical chatbot.

## Features

- **6 Deep Learning Models** — ResNet50, ResNet101, DenseNet121, DenseNet169, EfficientNet-B3, Vision Transformer (ViT)
- **Ensemble Prediction** — Simple averaging, weighted averaging, hard voting, stacking
- **Explainable AI** — Grad-CAM heatmaps (CNNs) and Attention Rollout (ViT)
- **Role-Based Access Control** — Patient, Doctor, Admin with enforced route protection
- **Doctor Review Workflow** — Doctors review AI predictions and submit clinical decisions
- **Email Notifications** — Patients notified via email when doctor submits a review
- **In-App Notifications** — Real-time notification bell for patients
- **AI Chatbot (ILMU)** — Powered by Claude Haiku for stroke education and system guidance
- **Google OAuth** — Sign in / sign up with Google
- **PDF Export** — Patients export full prediction history as a formatted PDF report
- **Audit Logging** — All system actions logged to Firestore
- **Security** — Flask-Talisman CSP, rate limiting, magic byte file validation, HttpOnly cookies, session timeout

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask, Python 3.11 |
| Frontend | Jinja2, Vanilla JS, CSS |
| Database | Firebase Firestore |
| Auth | Firebase Authentication (Email + Google OAuth) |
| ML | PyTorch, timm, HuggingFace Transformers |
| XAI | Grad-CAM, Attention Rollout |
| AI Chatbot | Anthropic Claude Haiku |
| Email | Resend API |
| Security | Flask-Talisman, Flask-Limiter, Flask-Login, Flask-Bcrypt |
| Deployment | Docker, Hugging Face Spaces |

## Live Demo

👉 **https://dancinggdogg-neuroscan.hf.space**

> Note: First load may take 5–10 minutes as model weights (~697 MB) are downloaded on cold start.

## Model Performance

Evaluated on a held-out test set of 34 MRI scans from Hospital Pengajar UPM (HPUPM).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| ResNet50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ResNet101 | 0.9706 | 0.9722 | 0.9706 | 0.9706 | 0.9827 |
| DenseNet121 | 0.9118 | 0.9167 | 0.9118 | 0.9118 | 0.9965 |
| DenseNet169 | 0.9412 | 0.9444 | 0.9412 | 0.9410 | 0.9970 |
| EfficientNet-B3 | 0.9706 | 0.9722 | 0.9706 | 0.9706 | 1.000 |
| Vision Transformer | 0.8824 | 0.8873 | 0.8824 | 0.8807 | 1.000 |
| **Ensemble (Best)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

## Local Setup

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
# Edit .env with your credentials
```

Required variables:
```
SECRET_KEY=
FLASK_DEBUG=true
FLASK_ENV=development
FIREBASE_API_KEY=
FIREBASE_AUTH_DOMAIN=
FIREBASE_PROJECT_ID=
ANTHROPIC_API_KEY=
RESEND_API_KEY=
RESEND_TO_EMAIL=
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
│   │   ├── model_loader.py  # Model loading + ensemble inference
│   │   ├── gradcam.py       # Grad-CAM implementation
│   │   └── vit_rollout.py   # ViT Attention Rollout
│   ├── static/              # CSS, JS, favicon
│   ├── templates/           # Jinja2 HTML templates
│   ├── routes.py            # All Flask routes
│   ├── models.py            # User model
│   └── __init__.py          # App factory + extensions
├── Dockerfile
├── .env.example
├── requirements.txt
└── run.py
```

## Known Limitations

| Limitation | Details |
|------------|---------|
| Email delivery on HF | Outbound SMTP is blocked on Hugging Face free tier. Email notifications are delivered via Resend API. On the free tier without a verified custom domain, emails are routed to the system administrator for demonstration purposes. Direct patient delivery works fully in local development. |
| Ephemeral GradCAM images | MRI heatmap images are lost on container restart. Persistent storage requires Hugging Face Pro or Firebase Storage. |
| Cold start delay | Model weights (~697 MB) are re-downloaded on every container restart — takes 5–10 minutes. |
| CPU-only inference | Free tier has no GPU. Predictions are slower than local inference. |
| CSRF disabled | CSRF protection is disabled due to incompatibility with Firebase JS SDK authentication flow. Risk is mitigated by `SameSite=Lax` cookie policy and Firebase ID token verification. |

## Developer

**Lee Ding Kuan**
UTAR, Faculty of Information and Communication Technology (Kampar) · 2025
Supervisor: Puan Nur Lyana Shahfiqa Bt Albashah