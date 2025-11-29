# GERD & Polyp Endoscopy Classification (ConvNeXt-Tiny, MLOps Pipeline)

This project implements a **complete MLOps workflow** for endoscopic image classification using the **ConvNeXt-Tiny** architecture.  
It detects **four gastrointestinal conditions**:

- GERD  
- GERD Normal  
- Polyp  
- Polyp Normal  

The project features:

- **Reproducible training pipeline** (DVC + MLflow)
- **Industrial-grade folder structure**
- **Config-driven experimentation (params.yaml)**
- **Production-ready inference API (FastAPI + Docker)**
- **GPU-accelerated training (Colab-compatible)**

This repository is designed for **research, production deployment, and portfolio demonstration**.

---

# Project Structure
```
gerdpolyp-mlops/
│
├── app/ # Production API
│ ├── main.py # FastAPI endpoint
│ ├── infer.py # Inference engine
│ ├── classes.json # Class mapping
│ └── init.py
│
├── artifacts/ # Output artifacts (model, curves, CM)
│ └── model_best.pth
│
├── data/
│ └── raw/ # Versioned dataset via DVC
│ ├── original/
│ └── augmented/
│
├── src/
│ ├── model/ # Training scripts
│ │ ├── train.py
│ │ ├── evaluate.py # (optional)
│ │ └── infer.py # Dev inference
│ ├── data/
│ │ └── dataset_loader.py
│ ├── utils/
│ │ └── load_config.py
│ └── config/
│ └── params.yaml
│
├── dvc.yaml # Training pipeline definition
├── dvc.lock # Pipeline lock for reproducibility
├── params.yaml # Root config (for DVC)
├── requirements.txt
├── Dockerfile
└── README.md
```

# ⚙️ MLOps Workflow Overview

This project uses a **three-phase MLOps pipeline**:

## **1. Experiment Management (MLflow)**
- Logs metrics per epoch  
- Logs parameters (batch size, learning rate, augmentation, etc.)  
- Stores confusion matrix & training history  
- Can be launched via:

```bash
mlflow ui --backend-store-uri mlruns

```

## **2. Data & Pipeline Versioning (DVC)**

* Dataset located in `data/raw/`
* Version-controlled via DVC
* Pipeline defined in `dvc.yaml`
* Re-run reproducible training:

```bash
dvc repro
```

# **3. Model Serving (FastAPI + Docker)**

* Production-grade inference engine
* Single-time model loading in memory
* Ready for deployment to Railway / Render / Fly.io / AWS / GCP

---

# Training & Reproducing Pipeline

## **Run training (local terminal)**

```bash
dvc repro
```

OR manually:

```bash
python src/model/train.py --config src/config/params.yaml
```

Training outputs:

```
artifacts/
├── model_best.pth
├── history.csv
└── confusion_matrix.png
```

---
# 🔮 Inference (Production)

## **Run API locally**

Install dependencies:

```bash
pip install -r requirements.txt
```

Start server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```
http://localhost:8000/docs
```

---
# 🐳 Docker Deployment

Build image:

```bash
docker build -t gerdpolyp-api .
```

Run container:

```bash
docker run -p 8000:8000 gerdpolyp-api
```

---
# 📚 Dataset Information

Dataset used: **GastroEndoNet**
Contains 4 classes of endoscopic images:

* GERD
* GERD Normal
* Polyp
* Polyp Normal

Augmented subset included.

---

# 🧠 Model Architecture

**ConvNeXt-Tiny**, fine-tuned:

* Pretrained: ImageNet-1K
* Patchify + ConvNeXt blocks
* Modified classifier (4 classes)
* Full fine-tuning

---

# 🔑 Key Metrics (Baseline Best)

| Metric              | Score                        |
| ------------------- | ---------------------------- |
| Accuracy            | ~0.90                        |
| Validation Accuracy | up to **90.48%**             |
| Precision           | High macro precision         |
| Recall              | Stable recall across classes |
| F1-score            | High macro F1 (~0.89–0.90)   |

---

# 🚀 Highlights of This Project

* Fully reproducible MLOps pipeline
* Industrial folder structure
* Seamless GPU/Colab compatibility
* Model serving API ready for production
* Docker-ready deployment
* Clean code, modular design

---

# 🤝 Contributing

Feel free to open PR, issues, or suggestions.

---

# 📄 License

MIT License.