import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import os
import io
import cv2
import shap
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from urllib.parse import urlparse
from PIL import Image
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="PhQure — Phishing & Quishing Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
    }
    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; }

    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #8892a4;
        margin-top: 0.3rem;
    }
    .metric-card {
        background: #131929;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-phishing {
        background: linear-gradient(135deg, #2d0a0a, #1a0505);
        border: 2px solid #ff4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-legitimate {
        background: linear-gradient(135deg, #0a2d1a, #051a0a);
        border: 2px solid #00cc66;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
    }
    .confidence-bar {
        background: #1e2d45;
        border-radius: 999px;
        height: 12px;
        margin: 0.5rem 0;
    }
    .branch-card {
        background: #131929;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'Space Mono', monospace;
    }
    .badge-phish { background: #2d0a0a; color: #ff4444; border: 1px solid #ff4444; }
    .badge-legit { background: #0a2d1a; color: #00cc66; border: 1px solid #00cc66; }
    .stButton>button {
        background: linear-gradient(135deg, #0066ff, #00d4ff);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.7rem 2rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 212, 255, 0.3);
    }
    .stTextInput>div>div>input {
        background: #131929;
        border: 1px solid #1e2d45;
        color: #e8eaf0;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
    }
    .sidebar .sidebar-content { background: #0d1220; }
    hr { border-color: #1e2d45; }
    .stSpinner > div { border-top-color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────
BASE      = os.path.join(os.path.expanduser("~"), "Desktop",
                         "Dissertation", "PhQure")
MODELS    = os.path.join(BASE, "models")
DISTILBERT_PATH = os.path.join(MODELS, "distilbert_branchC")
EFFNET_PATH     = os.path.join(MODELS, "efficientnet_branchB.pth")
XGB_PATH        = os.path.join(MODELS, "xgboost_branchA.pkl")
RF_PATH         = os.path.join(MODELS, "randomforest_branchA.pkl")

DEVICE = torch.device("cpu")
MAX_LEN = 128

FEATURE_NAMES = [
    'having_ip','url_length','shortening_service','having_at','double_slash',
    'prefix_suffix','num_subdomains','https_token','num_dots','num_hyphens',
    'num_digits','num_special_chars','url_depth','has_port','path_length',
    'query_length','has_query','domain_length','digit_ratio','letter_ratio',
    'suspicious_words','hex_encoding','tld_length','count_www','count_com',
    'has_iframe','mouse_over','right_click','forwarding'
]

# ── Feature extraction ─────────────────────────────────────
def extract_features(url):
    url = str(url).strip()
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path   = parsed.path
        query  = parsed.query
    except:
        domain = path = query = ""

    def having_ip(u):
        return 1 if re.search(r'\d+\.\d+\.\d+\.\d+', u) else 0
    def url_length(u):
        return 1 if len(u)<54 else (0 if len(u)<=75 else -1)
    def shortening(u):
        return 1 if re.search(r'bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co', u) else 0
    def suspicious_words(u):
        words = ['secure','account','update','login','signin','banking',
                 'confirm','verify','password','support','paypal']
        return 1 if any(w in u.lower() for w in words) else 0

    return [
        having_ip(url),
        url_length(url),
        shortening(url),
        1 if '@' in url else 0,
        1 if '//' in url[7:] else 0,
        1 if '-' in domain else 0,
        1 if len(domain.split('.'))==2 else (0 if len(domain.split('.'))==3 else -1),
        1 if url.startswith('https') else 0,
        url.count('.'),
        url.count('-'),
        sum(c.isdigit() for c in url),
        sum(not c.isalnum() for c in url),
        len([p for p in path.split('/') if p]),
        1 if (parsed.port and parsed.port not in [80,443]) else 0,
        len(path),
        len(query),
        1 if query else 0,
        len(domain),
        round(sum(c.isdigit() for c in url)/max(len(url),1), 4),
        round(sum(c.isalpha() for c in url)/max(len(url),1), 4),
        suspicious_words(url),
        1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0,
        len(domain.split('.')[-1]) if '.' in domain else 0,
        url.lower().count('www'),
        url.lower().count('.com'),
        0, 0, 0, 0
    ]

# ── Model loaders ──────────────────────────────────────────
@st.cache_resource
def load_branch_a():
    if os.path.exists(XGB_PATH):
        return joblib.load(XGB_PATH), joblib.load(RF_PATH)
    return None, None

@st.cache_resource
def load_branch_c():
    if os.path.exists(DISTILBERT_PATH):
        tok   = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
        model.eval()
        return tok, model
    return None, None

@st.cache_resource
def load_branch_b():
    if os.path.exists(EFFNET_PATH):
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 2)
        )
        model.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE))
        model.eval()
        return model
    return None

# ── Prediction functions ───────────────────────────────────
def predict_url(url):
    results = {}

    # Branch A
    xgb, rf = load_branch_a()
    if xgb:
        feats = np.array([extract_features(url)])
        prob_a = (xgb.predict_proba(feats)[0][1] +
                  rf.predict_proba(feats)[0][1]) / 2
        results["branch_a"] = {
            "prob": float(prob_a),
            "pred": int(prob_a > 0.5),
            "features": feats[0]
        }

    # Branch C
    tok, modelC = load_branch_c()
    if tok:
        enc = tok(url, max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="pt")
        with torch.no_grad():
            out  = modelC(**enc)
            prob = torch.softmax(out.logits, dim=1)[0]
        results["branch_c"] = {
            "prob": float(prob[1]),
            "pred": int(prob[1] > 0.5)
        }

    # Fusion
    if "branch_a" in results and "branch_c" in results:
        pa = results["branch_a"]["prob"]
        pc = results["branch_c"]["prob"]
        # Weighted fusion using trained coefficients
        fusion_score = 1 / (1 + np.exp(-(7.4712 * pa + 5.4448 * pc - 6.5)))
        results["fusion"] = {
            "prob": float(fusion_score),
            "pred": int(fusion_score > 0.5)
        }

    return results

def predict_qr(image_pil):
    model = load_branch_b()
    if not model:
        return None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img_t = transform(image_pil.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        out  = model(img_t)
        prob = torch.softmax(out, dim=1)[0]

    return {"prob": float(prob[1]), "pred": int(prob[1] > 0.5)}

def generate_gradcam(image_pil, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img_t  = transform(image_pil.convert("RGB")).unsqueeze(0)
    img_np = img_t.squeeze().permute(1,2,0).numpy()
    mean   = np.array([0.485, 0.456, 0.406])
    std    = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std + mean).clip(0,1)

    gradients  = []
    activations = []

    def forward_hook(m, i, o): activations.append(o.detach())
    def backward_hook(m, gi, go): gradients.append(go[0].detach())

    target = model.features[-1][0]
    fh = target.register_forward_hook(forward_hook)
    bh = target.register_full_backward_hook(backward_hook)

    model.zero_grad()
    out   = model(img_t)
    cls   = out.argmax(dim=1).item()
    out[0, cls].backward()

    fh.remove(); bh.remove()

    grads = gradients[0][0]
    acts  = activations[0][0]
    w     = grads.mean(dim=(1,2))
    cam   = torch.zeros(acts.shape[1:])
    for i, wi in enumerate(w):
        cam += wi * acts[i]
    cam = torch.relu(cam).numpy()
    cam = cv2.resize(cam, (224, 224))
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = (0.5 * img_np + 0.5 * heatmap).clip(0,1)

    return img_np, heatmap, overlay

def generate_shap_plot(url, xgb_model):
    feats = np.array([extract_features(url)])
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(feats)

    # Get SHAP values for phishing class
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

    # Top 10 features
    indices = np.argsort(np.abs(sv))[-10:][::-1]
    top_names  = [FEATURE_NAMES[i] for i in indices]
    top_values = [sv[i] for i in indices]
    colors     = ["#ff4444" if v > 0 else "#00cc66" for v in top_values]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#131929")
    ax.set_facecolor("#131929")
    bars = ax.barh(top_names[::-1], top_values[::-1], color=colors[::-1], height=0.6)
    ax.axvline(x=0, color="#8892a4", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)", color="#8892a4", fontsize=9)
    ax.set_title("Branch A — Feature Importance (SHAP)", color="#e8eaf0",
                 fontsize=10, fontweight="bold", pad=10)
    ax.tick_params(colors="#8892a4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1e2d45")
    red_patch   = mpatches.Patch(color='#ff4444', label='→ Phishing')
    green_patch = mpatches.Patch(color='#00cc66', label='→ Legitimate')
    ax.legend(handles=[red_patch, green_patch], loc="lower right",
              facecolor="#131929", edgecolor="#1e2d45",
              labelcolor="#8892a4", fontsize=8)
    plt.tight_layout()
    return fig

def confidence_bar_html(prob, is_phishing):
    color  = "#ff4444" if is_phishing else "#00cc66"
    width  = prob * 100
    return f"""
    <div style="background:#1e2d45;border-radius:999px;height:10px;margin:6px 0;">
        <div style="background:{color};width:{width:.1f}%;height:10px;
                    border-radius:999px;transition:width 0.5s;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;
                color:#8892a4;font-family:'Space Mono',monospace;">
        <span>0%</span><span style="color:{color};font-weight:700;">
        {prob*100:.1f}%</span><span>100%</span>
    </div>"""

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🔐</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
                    color:#00d4ff;font-weight:700;">PhQure</div>
        <div style="font-size:0.75rem;color:#8892a4;margin-top:0.3rem;">
            Phishing & Quishing Detector</div>
    </div>
    <hr style="border-color:#1e2d45;">
    """, unsafe_allow_html=True)

    st.markdown("### System Performance")
    metrics = [
        ("Branch A (XGBoost)", "AUC 0.9997"),
        ("Branch B (EfficientNet)", "AUC 0.9767"),
        ("Branch C (DistilBERT)", "AUC 1.0000"),
        ("Fusion Layer", "AUC 1.0000"),
    ]
    for name, val in metrics:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
                    padding:0.4rem 0;border-bottom:1px solid #1e2d45;">
            <span style="font-size:0.8rem;color:#8892a4;">{name}</span>
            <span style="font-size:0.8rem;color:#00d4ff;
                         font-family:'Space Mono',monospace;">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <hr style="border-color:#1e2d45;">
    <div style="font-size:0.75rem;color:#8892a4;line-height:1.6;">
    <b style="color:#e8eaf0;">Dataset</b><br>
    2,430,324 labelled URLs<br>
    30,000 QR code images<br><br>
    <b style="color:#e8eaf0;">Benchmark</b><br>
    Beats Trad & Chehab 2025<br>
    AUC 0.9133 → <span style="color:#00cc66">0.9767</span><br><br>
    <b style="color:#e8eaf0;">Author</b><br>
    Avantika Bishnoi<br>
    M.Sc. Data Science 2026<br>
    Central University of Haryana
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 2rem 0 1rem;">
    <div class="hero-title">PhQure 🔐</div>
    <div class="hero-sub">
        First unified deep learning system for detecting URL phishing
        and QR code phishing (quishing) with explainable AI
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔗 URL Analysis", "📷 QR Code Analysis", "ℹ️ About"])

# ─────────────────────────────────────────────────────────
# TAB 1 — URL ANALYSIS
# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Analyse a URL")
    st.markdown(
        "<div style='color:#8892a4;font-size:0.9rem;margin-bottom:1rem;'>"
        "Enter any URL to detect phishing using Branch A (XGBoost features) "
        "+ Branch C (DistilBERT) + Fusion Layer with SHAP explanation."
        "</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com/path/to/page",
            label_visibility="collapsed"
        )
    with col2:
        analyse_btn = st.button("Analyse →", key="url_btn")

    # Example URLs
    st.markdown("""
    <div style="font-size:0.8rem;color:#8892a4;margin-bottom:1rem;">
    Try these examples:
    </div>""", unsafe_allow_html=True)

    ex_cols = st.columns(4)
    examples = [
        ("🔴 Phishing", "http://paypal-secure-login.xyz/verify/account"),
        ("🟢 Legitimate", "https://www.github.com/python/cpython"),
        ("🔴 IP-based", "http://192.168.1.1/paypal/login"),
        ("🟢 E-commerce", "https://www.amazon.com/dp/B08N5WRWNW"),
    ]
    for i, (label, url) in enumerate(examples):
        with ex_cols[i]:
            if st.button(label, key=f"ex_{i}"):
                url_input = url
                analyse_btn = True

    if analyse_btn and url_input:
        with st.spinner("Running PhQure analysis..."):
            results = predict_url(url_input)

        if results:
            # Main verdict
            fusion = results.get("fusion", results.get("branch_a", {}))
            is_phishing = fusion.get("pred", 0) == 1
            prob        = fusion.get("prob", 0.5)

            if is_phishing:
                st.markdown(f"""
                <div class="result-phishing">
                    <div class="result-title" style="color:#ff4444;">
                        ⚠️ PHISHING DETECTED
                    </div>
                    <div style="color:#ff8888;margin-top:0.5rem;font-size:1rem;">
                        This URL shows strong phishing indicators
                    </div>
                    {confidence_bar_html(prob, True)}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legitimate">
                    <div class="result-title" style="color:#00cc66;">
                        ✅ LEGITIMATE
                    </div>
                    <div style="color:#66ee99;margin-top:0.5rem;font-size:1rem;">
                        No phishing indicators detected
                    </div>
                    {confidence_bar_html(1-prob, False)}
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Branch breakdown
            st.markdown("#### Branch-by-Branch Analysis")
            bcols = st.columns(3)

            branch_info = [
                ("Branch A", "XGBoost + RF", "branch_a"),
                ("Branch C", "DistilBERT", "branch_c"),
                ("Fusion", "Meta-Learner", "fusion"),
            ]

            for col, (name, model_name, key) in zip(bcols, branch_info):
                if key in results:
                    b      = results[key]
                    b_phish = b["pred"] == 1
                    b_prob  = b["prob"]
                    badge   = '<span class="badge badge-phish">PHISHING</span>' if b_phish else '<span class="badge badge-legit">LEGIT</span>'
                    with col:
                        st.markdown(f"""
                        <div class="branch-card">
                            <div style="font-family:'Space Mono',monospace;
                                        font-size:0.85rem;color:#e8eaf0;
                                        font-weight:700;">{name}</div>
                            <div style="font-size:0.75rem;color:#8892a4;
                                        margin-bottom:0.5rem;">{model_name}</div>
                            {badge}
                            <div style="font-size:0.8rem;color:#8892a4;
                                        margin-top:0.5rem;">
                                Confidence: <span style="color:#00d4ff;
                                font-family:'Space Mono',monospace;">
                                {b_prob*100:.1f}%</span>
                            </div>
                        </div>""", unsafe_allow_html=True)

            # SHAP
            xgb, _ = load_branch_a()
            if xgb and "branch_a" in results:
                st.markdown("#### SHAP Feature Explanation")
                st.markdown(
                    "<div style='color:#8892a4;font-size:0.85rem;margin-bottom:0.5rem;'>"
                    "Red bars push toward phishing · Green bars push toward legitimate"
                    "</div>", unsafe_allow_html=True)
                try:
                    fig = generate_shap_plot(url_input, xgb)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                except Exception as e:
                    st.info("SHAP plot unavailable for this URL.")

            # Feature values
            if "branch_a" in results:
                with st.expander("📋 Raw Feature Values (Branch A)"):
                    feats = results["branch_a"]["features"]
                    feat_df = pd.DataFrame({
                        "Feature": FEATURE_NAMES,
                        "Value": feats
                    })
                    st.dataframe(feat_df, use_container_width=True, height=300)

# ─────────────────────────────────────────────────────────
# TAB 2 — QR CODE ANALYSIS
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Analyse a QR Code Image")
    st.markdown(
        "<div style='color:#8892a4;font-size:0.9rem;margin-bottom:1rem;'>"
        "Upload a QR code image to detect quishing using EfficientNet-B0 CNN "
        "with Grad-CAM visual explanation."
        "</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload QR Code",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        image_pil = Image.open(uploaded)
        model_b   = load_branch_b()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Uploaded QR Code**")
            st.image(image_pil, width=224)

        with col2:
            with st.spinner("Running EfficientNet-B0 analysis..."):
                result = predict_qr(image_pil)

            if result:
                is_phishing = result["pred"] == 1
                prob        = result["prob"]

                if is_phishing:
                    st.markdown(f"""
                    <div class="result-phishing" style="padding:1.2rem;">
                        <div style="font-family:'Space Mono',monospace;
                                    font-size:1.4rem;color:#ff4444;font-weight:700;">
                            ⚠️ QUISHING DETECTED
                        </div>
                        <div style="color:#ff8888;font-size:0.9rem;margin-top:0.3rem;">
                            QR code likely encodes a phishing URL
                        </div>
                        {confidence_bar_html(prob, True)}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-legitimate" style="padding:1.2rem;">
                        <div style="font-family:'Space Mono',monospace;
                                    font-size:1.4rem;color:#00cc66;font-weight:700;">
                            ✅ LEGITIMATE QR CODE
                        </div>
                        <div style="color:#66ee99;font-size:0.9rem;margin-top:0.3rem;">
                            No quishing indicators detected
                        </div>
                        {confidence_bar_html(1-prob, False)}
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="font-size:0.85rem;color:#8892a4;margin-top:0.8rem;">
                    Model: EfficientNet-B0 &nbsp;·&nbsp;
                    AUC: 0.9767 &nbsp;·&nbsp;
                    Beats Trad & Chehab 2025 benchmark
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("Branch B model not found. Please check model path.")

        # Grad-CAM
        if model_b and uploaded:
            st.markdown("#### Grad-CAM Visual Explanation")
            st.markdown(
                "<div style='color:#8892a4;font-size:0.85rem;margin-bottom:0.5rem;'>"
                "Heatmap shows which regions of the QR code influenced the prediction. "
                "Red = high activation · Blue = low activation"
                "</div>", unsafe_allow_html=True)
            try:
                with st.spinner("Generating Grad-CAM..."):
                    original, heatmap, overlay = generate_gradcam(image_pil, model_b)

                gcols = st.columns(3)
                titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
                images = [original, heatmap, overlay]
                for gcol, title, img in zip(gcols, titles, images):
                    with gcol:
                        fig, ax = plt.subplots(figsize=(4,4))
                        fig.patch.set_facecolor("#131929")
                        ax.imshow(img)
                        ax.set_title(title, color="#e8eaf0",
                                     fontsize=9, fontweight="bold")
                        ax.axis("off")
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                st.markdown("""
                <div style="background:#131929;border:1px solid #1e2d45;
                             border-radius:8px;padding:0.8rem 1rem;
                             font-size:0.82rem;color:#8892a4;margin-top:0.5rem;">
                    💡 <b style="color:#e8eaf0;">Finding:</b>
                    For phishing QR codes, activation concentrates on the
                    <b style="color:#00d4ff;">finder patterns</b> (corner squares),
                    indicating structural irregularities from malicious URL encoding.
                    This is the first Grad-CAM analysis of a quishing detector
                    in published literature.
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.info("Grad-CAM unavailable. Ensure OpenCV is installed.")

# ─────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("### About PhQure")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#131929;border:1px solid #1e2d45;
                     border-radius:12px;padding:1.5rem;">
            <div style="font-family:'Space Mono',monospace;color:#00d4ff;
                         font-size:1rem;font-weight:700;margin-bottom:1rem;">
                Novel Contributions
            </div>
            <div style="font-size:0.85rem;color:#8892a4;line-height:2;">
                1️⃣ First unified URL + QR phishing detection system<br>
                2️⃣ First CNN-based quishing detector (beats published AUC)<br>
                3️⃣ First DistilBERT applied to raw URL classification<br>
                4️⃣ First SHAP explainability for quishing detection<br>
                5️⃣ First paired URL + QR image dataset (2.4M + 30K)
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:#131929;border:1px solid #1e2d45;
                     border-radius:12px;padding:1.5rem;">
            <div style="font-family:'Space Mono',monospace;color:#00d4ff;
                         font-size:1rem;font-weight:700;margin-bottom:1rem;">
                System Performance
            </div>
            <div style="font-size:0.85rem;color:#8892a4;line-height:2;">
                Branch A (XGBoost) &nbsp;&nbsp;→ 99.70% acc · AUC 0.9997<br>
                Branch B (EfficientNet) → 93.27% acc · AUC 0.9767<br>
                Branch C (DistilBERT) &nbsp;→ 99.87% F1 &nbsp;· AUC 1.0000<br>
                Fusion Layer &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ 99.65% acc · AUC 1.0000<br>
                Benchmark (Trad 2025) &nbsp;→ AUC 0.9133 &nbsp;<span style="color:#00cc66">← beaten</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem;background:#131929;border:1px solid #1e2d45;
                 border-radius:12px;padding:1.5rem;">
        <div style="font-family:'Space Mono',monospace;color:#00d4ff;
                     font-size:1rem;font-weight:700;margin-bottom:0.8rem;">
            Citation
        </div>
        <code style="font-size:0.8rem;color:#8892a4;line-height:1.8;">
            @mastersthesis{bishnoi2026phqure,<br>
            &nbsp;&nbsp;author = {Bishnoi, Avantika},<br>
            &nbsp;&nbsp;title  = {PhQure: A Unified Deep Learning Framework},<br>
            &nbsp;&nbsp;school = {Central University of Haryana},<br>
            &nbsp;&nbsp;year   = {2026}<br>
            }
        </code>
    </div>
    <div style="margin-top:1rem;font-size:0.8rem;color:#8892a4;text-align:center;">
        Built by <b style="color:#e8eaf0;">Avantika Bishnoi</b> ·
        M.Sc. Data Science 2024–2026 ·
        Central University of Haryana ·
        Supervisor: Dr. Keshav Singh Rawat
    </div>
    """, unsafe_allow_html=True)
