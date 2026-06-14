# PhQure — Phishing & Quishing Detection with Explainable AI

> Detects both **URL phishing** and **QR code phishing (quishing)** using a three-branch deep learning architecture with SHAP and Grad-CAM explainability. Deployed as an interactive Streamlit app.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## What is PhQure?

Phishing attacks increasingly use QR codes to bypass traditional URL scanners — a technique called **quishing**. Most tools handle either URLs or QR codes, not both.

PhQure solves this by running three models in parallel and combining their outputs through a fusion layer, giving a single confident verdict on whether a URL or QR code is malicious.

---

## Demo

> 🚀 **[Live App — coming soon](#)**

| URL Analysis | QR Code Upload | Explainability |
|---|---|---|
| Paste any URL | Upload a QR image | SHAP + Grad-CAM |

---

## Architecture

Three branches run independently. Their probability outputs are fed into a meta-learner (Logistic Regression) that makes the final decision.

```
Input URL ──► Branch A (XGBoost + RF on 30 features) ──► probability
           │
           ► Branch B (EfficientNet-B0 on QR image)   ──► probability ──► Fusion ──► Verdict
           │
           └► Branch C (DistilBERT on raw URL text)   ──► probability
```

### Branch results

| Branch | Model | Input | Accuracy | AUC |
|--------|-------|-------|----------|-----|
| A | XGBoost + Random Forest | 30 extracted URL features | 99.70% | 0.9997 |
| B | EfficientNet-B0 | QR code image | 93.27% | 0.9767 |
| C | DistilBERT (fine-tuned) | Raw URL text | 99.87% F1 | 1.0000 |
| **Fusion** | Logistic Regression | Branch A + C probabilities | **99.65%** | **1.0000** |

---

## Explainability

Knowing *why* a model flagged something is as important as the flag itself.

**Branch A — SHAP**
Feature importance plots show which of the 30 URL features (domain length, special character count, use of IP address, etc.) pushed the prediction toward phishing or legitimate.

**Branch B — Grad-CAM**
Heatmaps overlaid on QR code images show which visual regions the CNN focused on. Phishing QR codes consistently activate on the finder pattern regions (corner markers).

**Branch C — DistilBERT attention**
Token-level attention shows which parts of the raw URL string the model weighted most heavily.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep learning | PyTorch, HuggingFace Transformers, EfficientNet |
| Classical ML | XGBoost, scikit-learn, Random Forest |
| Explainability | SHAP, Grad-CAM |
| App | Streamlit |
| Training | Google Colab (T4 GPU) |
| Data | SQLite (~2.4M URLs), 30,000 QR images |
| Language | Python 3.10 |

---

## Project Structure

```
PhQure/
├── notebooks/
│   ├── branch_a_xgboost.ipynb       # URL feature-based model
│   ├── branch_b_efficientnet.ipynb  # QR image CNN
│   ├── branch_c_distilbert.ipynb    # URL text NLP model
│   └── fusion_layer.ipynb           # Meta-learner + evaluation
│
├── src/
│   ├── feature_extraction.py        # 30 URL feature extractor
│   ├── train_branch_a.py
│   ├── train_branch_b.py
│   ├── train_branch_c.py
│   └── fusion.py
│
├── explainability/
│   ├── shap_branch_a.png            # SHAP summary plot
│   └── gradcam_branch_b.png         # Grad-CAM heatmap examples
│
├── models/                          # Weights saved to Google Drive
├── data/                            # Dataset info (not uploaded — too large)
├── app.py                           # Streamlit demo
└── requirements.txt
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Avantika029/PhQure.git
cd PhQure

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> **Note**: Model weights are not included in this repo due to file size. Download them from [Google Drive](#) and place in the `models/` folder.

---

## Dataset

- **URLs**: ~2.4 million labelled URLs stored in SQLite
- **QR codes**: 30,000 images split across phishing and legitimate classes
- Data was not uploaded to this repo due to size constraints

---

## Known Limitations

- Branch C (DistilBERT) shows a bias toward flagging URLs with long path strings as phishing, even when legitimate. This is a dataset characteristic, not a model bug — and the fusion layer partially corrects for it.
- Branch B accuracy (93.27%) is lower than the URL branches because visual QR code differences between phishing and legitimate are subtler than URL-level features.

---

## Skills Demonstrated

`Deep Learning` `NLP` `Computer Vision` `Ensemble Methods` `Explainable AI` `Feature Engineering` `Model Deployment` `Transfer Learning` `Fine-tuning` `Data Analysis`

---

## About

Built as part of M.Sc. Data Science dissertation — Central University of Haryana (2024–2026).

**Supervised by**: Dr. Keshav Singh Rawat

---

## License

MIT License — feel free to use, fork, and build on this.
