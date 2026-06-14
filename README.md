# \# PhQure — Phishing \& Quishing Detection with Explainable AI

# 

# A machine learning system that detects both \*\*URL phishing\*\* and 

# \*\*QR code phishing (quishing)\*\* using a three-branch architecture 

# with SHAP and Grad-CAM explainability.

# 

# Built as part of M.Sc. Data Science dissertation at 

# Central University of Haryana (2024–2026).

# 

# \---

# 

# \## What it does

# 

# \- Paste a URL → model tells you if it's phishing or legitimate

# \- Upload a QR code image → model scans and classifies it

# \- Shows \*why\* it made that decision using SHAP feature importance 

# &#x20; and Grad-CAM heatmaps

# 

# \---

# 

# \## Architecture

# 

# Three models run in parallel and their outputs are combined:

# 

# | Branch | Model | Input | Accuracy |

# |--------|-------|-------|----------|

# | A | XGBoost + Random Forest | 30 extracted URL features | 99.70% |

# | B | EfficientNet-B0 (CNN) | QR code image | 93.27% |

# | C | DistilBERT (fine-tuned) | Raw URL text | 99.87% F1 |

# | Fusion | Logistic Regression | Branch A + C probabilities | 99.65% |

# 

# \---

# 

# \## Explainability

# 

# \- \*\*Branch A\*\* — SHAP values show which URL features (length, 

# &#x20; special characters, domain age etc.) drove the prediction

# \- \*\*Branch B\*\* — Grad-CAM heatmaps highlight which parts of the 

# &#x20; QR code image the model focused on

# \- \*\*Branch C\*\* — DistilBERT attention over raw URL tokens

# 

# \---

# 

# \## Tech stack

# 

# \- Python, PyTorch, HuggingFace Transformers

# \- XGBoost, scikit-learn

# \- SHAP, Grad-CAM

# \- Streamlit (demo app)

# \- Google Colab (T4 GPU for training)

# \- SQLite (2.4M URL dataset)

# 

# \---

# 

# \## Project structure

# PhQure/

# 

# ├── notebooks/        # Training notebooks for each branch

# 

# ├── src/              # Python scripts

# 

# ├── explainability/   # SHAP plots and Grad-CAM heatmaps

# 

# ├── models/           # Model weights (stored on Google Drive)

# 

# ├── data/             # Dataset info (not uploaded due to size)

# 

# ├── app.py            # Streamlit demo

# 

# └── requirements.txt

# 

# \---

# 

# \## Run locally

# 

# ```bash

# git clone https://github.com/Avantika029/PhQure.git

# cd PhQure

# pip install -r requirements.txt

# streamlit run app.py

# ```

# 

# \---

# 

# \## Dataset

# 

# \- \*\*URLs\*\*: \~2.4 million labelled URLs (SQLite database)

# \- \*\*QR codes\*\*: 30,000 images (phishing + legitimate)

# 

# \---

# 

# \## Results

# 

# | Metric | Branch A | Branch B | Branch C | Fusion |

# |--------|----------|----------|----------|--------|

# | Accuracy | 99.70% | 93.27% | 99.87% | 99.65% |

# | AUC | 0.9997 | 0.9767 | 1.0000 | 1.0000 |

