import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import shap
import re
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from PIL import Image
from urllib.parse import urlparse

st.set_page_config(
    page_title="PhQure",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_models():
    models = {}
    models['xgb'] = joblib.load('models/branch_a_xgb.pkl')
    models['rf'] = joblib.load('models/branch_a_rf.pkl')
    models['fusion'] = joblib.load('models/fusion_meta_lr.pkl')
    models['tokenizer'] = DistilBertTokenizer.from_pretrained('bishnoiavantika1/phqure-distilbert-branchc')
    models['distilbert'] = DistilBertForSequenceClassification.from_pretrained('bishnoiavantika1/phqure-distilbert-branchc')
    models['distilbert'].eval()
    return models

models = load_models()

def extract_features(url):
    url = str(url).strip()
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    if len(url) < 54: ul = 1
    elif len(url) <= 75: ul = 0
    else: ul = -1

    domain_parts = domain.split('.') if domain else ['']
    if len(domain_parts) == 2: ns = 1
    elif len(domain_parts) == 3: ns = 0
    else: ns = -1

    features = {
        'having_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        'url_length': ul,
        'shortening_service': 1 if re.search(r'bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|tiny\.cc', url) else 0,
        'having_at': 1 if '@' in url else 0,
        'double_slash': 1 if '//' in url[7:] else 0,
        'prefix_suffix': 1 if '-' in domain else 0,
        'num_subdomains': ns,
        'https_token': 1 if url.startswith('https') else 0,
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special_chars': sum(not c.isalnum() for c in url),
        'url_depth': len([p for p in path.split('/') if p]),
        'has_port': 1 if parsed.port and parsed.port not in [80, 443] else 0,
        'path_length': len(path),
        'query_length': len(query),
        'has_query': 1 if query else 0,
        'domain_length': len(domain),
        'digit_ratio': round(sum(c.isdigit() for c in url) / max(len(url), 1), 4),
        'letter_ratio': round(sum(c.isalpha() for c in url) / max(len(url), 1), 4),
        'suspicious_words': 1 if any(w in url.lower() for w in ['secure','account','update','login','signin','banking','confirm','verify','password','support','paypal','ebay','amazon','apple','microsoft']) else 0,
        'hex_encoding': 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0,
        'tld_length': len(domain_parts[-1]) if domain else 0,
        'count_www': url.lower().count('www'),
        'count_com': url.lower().count('.com'),
        'has_iframe': 0,
        'mouse_over': 0,
        'right_click': 0,
        'forwarding': 0
    }
    return pd.DataFrame([features])

def predict_distilbert(url):
    tokenizer = models['tokenizer']
    model = models['distilbert']
    encoding = tokenizer(
        url,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**encoding)
        proba = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return proba

st.title("🛡️ PhQure — Phishing & Quishing Detection")
tab1, tab2, tab3 = st.tabs(["🔗 URL Analysis", "📷 QR Code Analysis", "ℹ️ About"])

with tab1:
    st.header("URL Phishing Detection")
    st.info("💡 Enter the full URL including path for best results e.g. example.com/login/verify")
    url_input = st.text_input("Enter a URL to analyse:", placeholder="example.com/page/login")

    if st.button("Analyse URL") and url_input:
        if '/' not in url_input.replace('://', ''):
            st.warning("⚠️ Domain-only URLs may not be accurate. Include the full path for best results.")

        with st.spinner("Analysing..."):
            features_df = extract_features(url_input)
            xgb_proba = models['xgb'].predict_proba(features_df)[0][1]
            c_proba = predict_distilbert(url_input)
            fusion_input = np.array([[xgb_proba, c_proba]])
            fusion_proba = models['fusion'].predict_proba(fusion_input)[0][1]

            st.subheader("Result")
            xgb_pred = 1 if xgb_proba > 0.5 else 0
            if xgb_pred == 1:
                st.error(f"⚠️ PHISHING DETECTED — Confidence: {xgb_proba*100:.1f}%")
            else:
                st.success(f"✅ LEGITIMATE — Confidence: {(1-xgb_proba)*100:.1f}%")

            st.subheader("Branch Scores")
            col1, col2, col3 = st.columns(3)
            col1.metric("Branch A (XGBoost)", f"{xgb_proba*100:.1f}%", "Phishing probability")
            col2.metric("Branch C (DistilBERT)", f"{c_proba*100:.1f}%", "Phishing probability")
            col3.metric("Fusion Score", f"{fusion_proba*100:.1f}%", "Final phishing probability")

            st.subheader("Feature Importance (SHAP)")
            explainer = shap.TreeExplainer(models['xgb'])
            shap_values = explainer.shap_values(features_df)
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=features_df.iloc[0],
                    feature_names=features_df.columns.tolist()
                ),
                show=False
            )
            st.pyplot(fig)
            plt.close()

with tab2:
    st.header("QR Code Quishing Detection")
    st.info("Upload a QR code image to detect if it leads to a phishing site.")
    uploaded_file = st.file_uploader("Upload QR Code Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded QR Code", width=300)
        st.info("Branch B (EfficientNet-B0) model requires GPU. For full analysis, run locally with the trained model.")

with tab3:
    st.header("About PhQure")
    st.markdown("""
    **PhQure** is a unified deep learning framework for detecting phishing URLs and QR code phishing (quishing).

    ### Architecture
    - **Branch A**: XGBoost + Random Forest on 29 extracted URL features
    - **Branch B**: EfficientNet-B0 CNN on QR code images
    - **Branch C**: DistilBERT fine-tuned on raw URL text
    - **Fusion Layer**: Logistic Regression meta-learner

    ### Results
    | Branch | Model | Accuracy | AUC |
    |--------|-------|----------|-----|
    | A | XGBoost + RF | 99.67% | 0.9997 |
    | B | EfficientNet-B0 | 92.77% | 0.9780 |
    | C | DistilBERT | 99.88% | 1.0000 |
    | Fusion | Logistic Regression | 99.65% | 0.9991 |

    ### Explainability
    - SHAP for URL branches
    - Grad-CAM for QR code branch

    ### Note
    For best results enter the full URL including path e.g. `example.com/page/login`
    """)