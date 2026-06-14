import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import os
import cv2
import shap
import joblib
import gdown
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from urllib.parse import urlparse
from PIL import Image
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="PhQure — Phishing & Quishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.main { background-color: #0a0e1a; }
.stApp { background-color: #0a0e1a; }
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3rem; font-weight: 700;
    color: #00d4ff; letter-spacing: -1px; margin-bottom: 0;
}
.hero-sub { font-size: 1.1rem; color: #8892a4; margin-top: 0.3rem; }
.result-phishing {
    background: linear-gradient(135deg, #2d0a0a, #1a0505);
    border: 2px solid #ff4444; border-radius: 16px;
    padding: 2rem; text-align: center;
}
.result-legitimate {
    background: linear-gradient(135deg, #0a2d1a, #051a0a);
    border: 2px solid #00cc66; border-radius: 16px;
    padding: 2rem; text-align: center;
}
.result-title { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; }
.branch-card {
    background: #131929; border: 1px solid #1e2d45;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.badge {
    display: inline-block; padding: 0.2rem 0.7rem;
    border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-phish { background: #2d0a0a; color: #ff4444; border: 1px solid #ff4444; }
.badge-legit { background: #0a2d1a; color: #00cc66; border: 1px solid #00cc66; }
.stButton>button {
    background: linear-gradient(135deg, #0066ff, #00d4ff);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace; font-weight: 700;
    font-size: 1rem; padding: 0.7rem 2rem; width: 100%;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,212,255,0.3); }
.stTextInput>div>div>input {
    background: #131929; border: 1px solid #1e2d45;
    color: #e8eaf0; border-radius: 8px;
}
hr { border-color: #1e2d45; }
</style>
""", unsafe_allow_html=True)

# ── Model paths & Drive IDs ─────────────────────────────────────
MODELS_DIR = "models"
DISTILBERT_DIR = os.path.join(MODELS_DIR, "distilbert_branchC")
os.makedirs(DISTILBERT_DIR, exist_ok=True)

# Google Drive file IDs
DRIVE_IDS = {
    os.path.join(MODELS_DIR, "xgboost_branchA.pkl")              : "1AlkSjWfpwv6h8_xzRBqg_X0S-kXFsTh6",
    os.path.join(MODELS_DIR, "randomforest_branchA.pkl")         : "1ObXYNmSy06QhRVxIsJz1j72bSvtEPR10",
    os.path.join(MODELS_DIR, "efficientnet_branchB.pth")         : "1D7tBHu4DfgG33VpwFvIl2v1ww_mXgXIc",
    os.path.join(DISTILBERT_DIR, "config.json")                  : "1Gb2TnV_OqK6bC3-Qz7loMkgwtPfWV4A1",
    os.path.join(DISTILBERT_DIR, "model.safetensors")            : "1paYuJcCjhj06fqqUOCd7EAYB9V3GA_KF",
    os.path.join(DISTILBERT_DIR, "tokenizer.json")               : "1s6Pydw2fF9eHbm5p23z9VJAcgTbO8uoX",
    os.path.join(DISTILBERT_DIR, "tokenizer_config.json")        : "1e1pYQXEfzkHcGY3ifKWp9vZ_z47FVn7i",
}

DEVICE  = torch.device("cpu")
MAX_LEN = 128

FEATURE_NAMES = [
    'having_ip','url_length','shortening_service','having_at','double_slash',
    'prefix_suffix','num_subdomains','https_token','num_dots','num_hyphens',
    'num_digits','num_special_chars','url_depth','has_port','path_length',
    'query_length','has_query','domain_length','digit_ratio','letter_ratio',
    'suspicious_words','hex_encoding','tld_length','count_www','count_com',
    'has_iframe','mouse_over','right_click','forwarding'
]

# ── Download models from Drive ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_models():
    for path, file_id in DRIVE_IDS.items():
        if not os.path.exists(path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=True)
    return True

# ── Feature extraction ──────────────────────────────────────────
def extract_features(url):
    url = str(url).strip()
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path   = parsed.path
        query  = parsed.query
    except:
        domain = path = query = ""

    def having_ip(u):      return 1 if re.search(r'\d+\.\d+\.\d+\.\d+', u) else 0
    def url_length(u):     return 1 if len(u)<54 else (0 if len(u)<=75 else -1)
    def shortening(u):     return 1 if re.search(r'bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co', u) else 0
    def susp_words(u):
        words = ['secure','account','update','login','signin','banking',
                 'confirm','verify','password','support','paypal']
        return 1 if any(w in u.lower() for w in words) else 0

    return [
        having_ip(url), url_length(url), shortening(url),
        1 if '@' in url else 0,
        1 if '//' in url[7:] else 0,
        1 if '-' in domain else 0,
        1 if len(domain.split('.'))==2 else (0 if len(domain.split('.'))==3 else -1),
        1 if url.startswith('https') else 0,
        url.count('.'), url.count('-'),
        sum(c.isdigit() for c in url),
        sum(not c.isalnum() for c in url),
        len([p for p in path.split('/') if p]),
        1 if (parsed.port and parsed.port not in [80,443]) else 0,
        len(path), len(query), 1 if query else 0, len(domain),
        round(sum(c.isdigit() for c in url)/max(len(url),1), 4),
        round(sum(c.isalpha() for c in url)/max(len(url),1), 4),
        susp_words(url),
        1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0,
        len(domain.split('.')[-1]) if '.' in domain else 0,
        url.lower().count('www'), url.lower().count('.com'),
        0, 0, 0, 0
    ]

# ── Model loaders ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_branch_a():
    xgb_path = os.path.join(MODELS_DIR, "xgboost_branchA.pkl")
    rf_path  = os.path.join(MODELS_DIR, "randomforest_branchA.pkl")
    if os.path.exists(xgb_path) and os.path.exists(rf_path):
        return joblib.load(xgb_path), joblib.load(rf_path)
    return None, None

@st.cache_resource(show_spinner=False)
def load_branch_b():
    path = os.path.join(MODELS_DIR, "efficientnet_branchB.pth")
    if os.path.exists(path):
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(m.classifier[1].in_features, 2)
        )
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        return m
    return None

@st.cache_resource(show_spinner=False)
def load_branch_c():
    if os.path.exists(DISTILBERT_DIR) and os.path.exists(
            os.path.join(DISTILBERT_DIR, "model.safetensors")):
        tok = DistilBertTokenizer.from_pretrained(DISTILBERT_DIR)
        mdl = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_DIR)
        mdl.eval()
        return tok, mdl
    return None, None

# ── Prediction ──────────────────────────────────────────────────
def predict_url(url):
    results = {}
    xgb, rf = load_branch_a()
    if xgb:
        feats  = np.array([extract_features(url)])
        prob_a = (xgb.predict_proba(feats)[0][1] + rf.predict_proba(feats)[0][1]) / 2
        results["branch_a"] = {"prob": float(prob_a), "pred": int(prob_a>0.5), "features": feats[0]}

    tok, mdlC = load_branch_c()
    if tok:
        enc = tok(url, max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="pt")
        with torch.no_grad():
            prob = torch.softmax(mdlC(**enc).logits, dim=1)[0]
        results["branch_c"] = {"prob": float(prob[1]), "pred": int(prob[1]>0.5)}

    if "branch_a" in results and "branch_c" in results:
        pa = results["branch_a"]["prob"]
        pc = results["branch_c"]["prob"]
        # Trained fusion coefficients: Branch A=7.4712, Branch C=5.4448, intercept=-6.2
        score = 1 / (1 + np.exp(-(7.4712*pa + 5.4448*pc - 6.2)))
        results["fusion"] = {"prob": float(score), "pred": int(score>0.5)}
    return results

def predict_qr(image_pil):
    mdl = load_branch_b()
    if not mdl:
        return None
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    with torch.no_grad():
        prob = torch.softmax(mdl(tf(image_pil.convert("RGB")).unsqueeze(0)), dim=1)[0]
    return {"prob": float(prob[1]), "pred": int(prob[1]>0.5)}

def generate_gradcam(image_pil, mdl):
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img_t  = tf(image_pil.convert("RGB")).unsqueeze(0)
    img_np = img_t.squeeze().permute(1,2,0).numpy()
    img_np = (img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])).clip(0,1)

    gradients, activations = [], []
    fh = mdl.features[-1][0].register_forward_hook(lambda m,i,o: activations.append(o.detach()))
    bh = mdl.features[-1][0].register_full_backward_hook(lambda m,gi,go: gradients.append(go[0].detach()))
    mdl.zero_grad()
    out = mdl(img_t)
    out[0, out.argmax(dim=1).item()].backward()
    fh.remove(); bh.remove()

    grads = gradients[0][0]; acts = activations[0][0]
    w   = grads.mean(dim=(1,2))
    cam = sum(w[i]*acts[i] for i in range(len(w)))
    cam = torch.relu(cam).numpy()
    cam = cv2.resize(cam, (224,224))
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
    return img_np, heatmap, (0.5*img_np + 0.5*heatmap).clip(0,1)

def generate_shap_plot(url, xgb_model):
    feats       = np.array([extract_features(url)])
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(feats)
    sv          = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    indices     = np.argsort(np.abs(sv))[-10:][::-1]
    top_names   = [FEATURE_NAMES[i] for i in indices]
    top_values  = [sv[i] for i in indices]
    colors      = ["#ff4444" if v>0 else "#00cc66" for v in top_values]

    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor("#131929"); ax.set_facecolor("#131929")
    ax.barh(top_names[::-1], top_values[::-1], color=colors[::-1], height=0.6)
    ax.axvline(x=0, color="#8892a4", linewidth=0.8)
    ax.set_xlabel("SHAP Value", color="#8892a4", fontsize=9)
    ax.set_title("Branch A — Feature Importance (SHAP)", color="#e8eaf0", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#8892a4", labelsize=8)
    for spine in ax.spines.values(): spine.set_color("#1e2d45")
    ax.legend(handles=[
        mpatches.Patch(color='#ff4444', label='→ Phishing'),
        mpatches.Patch(color='#00cc66', label='→ Legitimate')
    ], loc="lower right", facecolor="#131929", edgecolor="#1e2d45", labelcolor="#8892a4", fontsize=8)
    plt.tight_layout()
    return fig

def confidence_bar_html(prob, is_phishing):
    color = "#ff4444" if is_phishing else "#00cc66"
    return f"""
    <div style="background:#1e2d45;border-radius:999px;height:10px;margin:6px 0;">
        <div style="background:{color};width:{prob*100:.1f}%;height:10px;border-radius:999px;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#8892a4;font-family:'Space Mono',monospace;">
        <span>0%</span><span style="color:{color};font-weight:700;">{prob*100:.1f}%</span><span>100%</span>
    </div>"""

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🛡️</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#00d4ff;font-weight:700;">PhQure</div>
        <div style="font-size:0.75rem;color:#8892a4;margin-top:0.3rem;">Phishing & Quishing Detector</div>
    </div><hr style="border-color:#1e2d45;">
    """, unsafe_allow_html=True)

    st.markdown("### System Performance")
    for name, val in [
        ("Branch A (XGBoost+RF)", "AUC 0.9997"),
        ("Branch B (EfficientNet)", "AUC 0.9767"),
        ("Branch C (DistilBERT)", "AUC 1.0000"),
        ("Fusion Layer", "99.65% acc"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid #1e2d45;">
            <span style="font-size:0.8rem;color:#8892a4;">{name}</span>
            <span style="font-size:0.8rem;color:#00d4ff;font-family:'Space Mono',monospace;">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <hr style="border-color:#1e2d45;">
    <div style="font-size:0.75rem;color:#8892a4;line-height:1.8;">
        <b style="color:#e8eaf0;">Dataset</b><br>
        2.4M labelled URLs<br>
        30,000 QR code images<br><br>
        <b style="color:#e8eaf0;">Stack</b><br>
        PyTorch · HuggingFace · XGBoost<br>
        SHAP · Grad-CAM · Streamlit<br><br>
        <b style="color:#e8eaf0;">Author</b><br>
        Avantika Bishnoi<br>
        M.Sc. Data Science 2026<br>
        Central University of Haryana
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 0 1rem;">
    <div class="hero-title">PhQure 🛡️</div>
    <div class="hero-sub">Dual-threat phishing detector — URLs and QR codes, with explainable AI</div>
</div>
""", unsafe_allow_html=True)

# ── Download models on startup ───────────────────────────────────
with st.spinner("Loading models (first run may take 1-2 minutes)..."):
    download_models()

tab1, tab2, tab3 = st.tabs(["🔗 URL Analysis", "📷 QR Code Analysis", "ℹ️ About"])

# ────────────────────────────────────────────────────────────────
# TAB 1 — URL ANALYSIS
# ────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Analyse a URL")
    st.markdown("<div style='color:#8892a4;font-size:0.9rem;margin-bottom:1rem;'>Enter any URL to check for phishing using three models + fusion layer with SHAP explanation.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([4,1])
    with col1:
        url_input = st.text_input("URL", placeholder="https://example.com", label_visibility="collapsed")
    with col2:
        analyse_btn = st.button("Analyse →", key="url_btn")

    st.markdown("<div style='font-size:0.8rem;color:#8892a4;margin-bottom:0.5rem;'>Try an example:</div>", unsafe_allow_html=True)
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
            fusion      = results.get("fusion", results.get("branch_a", {}))
            is_phishing = fusion.get("pred", 0) == 1
            prob        = fusion.get("prob", 0.5)

            if is_phishing:
                st.markdown(f"""
                <div class="result-phishing">
                    <div class="result-title" style="color:#ff4444;">⚠️ PHISHING DETECTED</div>
                    <div style="color:#ff8888;margin-top:0.5rem;">This URL shows strong phishing indicators</div>
                    {confidence_bar_html(prob, True)}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legitimate">
                    <div class="result-title" style="color:#00cc66;">✅ LEGITIMATE</div>
                    <div style="color:#66ee99;margin-top:0.5rem;">No phishing indicators detected</div>
                    {confidence_bar_html(1-prob, False)}
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Branch-by-Branch Breakdown")
            bcols = st.columns(3)
            for col, (name, mname, key) in zip(bcols, [
                ("Branch A", "XGBoost + RF", "branch_a"),
                ("Branch C", "DistilBERT",   "branch_c"),
                ("Fusion",   "Meta-Learner", "fusion"),
            ]):
                if key in results:
                    b = results[key]
                    badge = '<span class="badge badge-phish">PHISHING</span>' if b["pred"]==1 else '<span class="badge badge-legit">LEGIT</span>'
                    with col:
                        st.markdown(f"""
                        <div class="branch-card">
                            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#e8eaf0;font-weight:700;">{name}</div>
                            <div style="font-size:0.75rem;color:#8892a4;margin-bottom:0.5rem;">{mname}</div>
                            {badge}
                            <div style="font-size:0.8rem;color:#8892a4;margin-top:0.5rem;">
                                Confidence: <span style="color:#00d4ff;font-family:'Space Mono',monospace;">{b['prob']*100:.1f}%</span>
                            </div>
                        </div>""", unsafe_allow_html=True)

            xgb, _ = load_branch_a()
            if xgb and "branch_a" in results:
                st.markdown("#### SHAP Feature Explanation")
                st.markdown("<div style='color:#8892a4;font-size:0.85rem;margin-bottom:0.5rem;'>Red → phishing · Green → legitimate</div>", unsafe_allow_html=True)
                try:
                    fig = generate_shap_plot(url_input, xgb)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                except:
                    st.info("SHAP unavailable for this URL.")

            if "branch_a" in results:
                with st.expander("📋 Raw Feature Values (Branch A)"):
                    st.dataframe(pd.DataFrame({"Feature": FEATURE_NAMES, "Value": results["branch_a"]["features"]}), use_container_width=True, height=300)

# ────────────────────────────────────────────────────────────────
# TAB 2 — QR CODE ANALYSIS
# ────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Analyse a QR Code")
    st.markdown("<div style='color:#8892a4;font-size:0.9rem;margin-bottom:1rem;'>Upload a QR image or scan one live with your camera. EfficientNet-B0 classifies it with Grad-CAM explanation.</div>", unsafe_allow_html=True)

    input_method = st.radio("Input method", ["📁 Upload image", "📷 Scan with camera"], horizontal=True)

    image_pil = None

    if input_method == "📁 Upload image":
        uploaded = st.file_uploader("Upload QR Code", type=["png","jpg","jpeg"], label_visibility="collapsed")
        if uploaded:
            image_pil = Image.open(uploaded)

    else:
        st.markdown("<div style='color:#8892a4;font-size:0.85rem;margin-bottom:0.5rem;'>Point your camera at a QR code and click the capture button.</div>", unsafe_allow_html=True)
        camera_photo = st.camera_input("Scan QR Code", label_visibility="collapsed")
        if camera_photo:
            image_pil = Image.open(camera_photo)

    if image_pil:
        model_b = load_branch_b()
        col1, col2 = st.columns([1,2])

        with col1:
            st.markdown("**QR Code**")
            st.image(image_pil, width=224)

        with col2:
            with st.spinner("Running EfficientNet-B0..."):
                result = predict_qr(image_pil)

            if result:
                is_phishing = result["pred"] == 1
                prob        = result["prob"]
                if is_phishing:
                    st.markdown(f"""
                    <div class="result-phishing" style="padding:1.2rem;">
                        <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#ff4444;font-weight:700;">⚠️ QUISHING DETECTED</div>
                        <div style="color:#ff8888;font-size:0.9rem;margin-top:0.3rem;">QR code likely encodes a phishing URL</div>
                        {confidence_bar_html(prob, True)}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-legitimate" style="padding:1.2rem;">
                        <div style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#00cc66;font-weight:700;">✅ LEGITIMATE QR CODE</div>
                        <div style="color:#66ee99;font-size:0.9rem;margin-top:0.3rem;">No quishing indicators detected</div>
                        {confidence_bar_html(1-prob, False)}
                    </div>""", unsafe_allow_html=True)
                st.markdown("<div style='font-size:0.8rem;color:#8892a4;margin-top:0.8rem;'>Model: EfficientNet-B0 · AUC: 0.9767</div>", unsafe_allow_html=True)
            else:
                st.warning("Branch B model not loaded yet.")

        if model_b:
            st.markdown("#### Grad-CAM Visual Explanation")
            st.markdown("<div style='color:#8892a4;font-size:0.85rem;margin-bottom:0.5rem;'>Heatmap shows which QR regions influenced the prediction. Red = high activation · Blue = low.</div>", unsafe_allow_html=True)
            try:
                with st.spinner("Generating Grad-CAM..."):
                    original, heatmap, overlay = generate_gradcam(image_pil, model_b)
                gcols = st.columns(3)
                for gcol, title, img in zip(gcols, ["Original","Grad-CAM Heatmap","Overlay"], [original, heatmap, overlay]):
                    with gcol:
                        fig, ax = plt.subplots(figsize=(4,4))
                        fig.patch.set_facecolor("#131929")
                        ax.imshow(img); ax.set_title(title, color="#e8eaf0", fontsize=9, fontweight="bold"); ax.axis("off")
                        st.pyplot(fig, use_container_width=True); plt.close()
                st.markdown("""
                <div style="background:#131929;border:1px solid #1e2d45;border-radius:8px;padding:0.8rem 1rem;font-size:0.82rem;color:#8892a4;margin-top:0.5rem;">
                    🔍 <b style="color:#e8eaf0;">Finding:</b> For phishing QR codes, Grad-CAM activation concentrates on the
                    <b style="color:#00d4ff;">finder patterns</b> (corner squares), indicating structural irregularities from malicious URL encoding.
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.info(f"Grad-CAM unavailable: {e}")

# ────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### About PhQure")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#131929;border:1px solid #1e2d45;border-radius:12px;padding:1.5rem;">
            <div style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:1rem;font-weight:700;margin-bottom:1rem;">What it does</div>
            <div style="font-size:0.85rem;color:#8892a4;line-height:2;">
                ✅ Detects phishing URLs using 3 independent models<br>
                ✅ Detects quishing (QR code phishing) via CNN<br>
                ✅ Explains every prediction with SHAP + Grad-CAM<br>
                ✅ Supports live camera QR scanning<br>
                ✅ Fuses all branches for a final confident verdict
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#131929;border:1px solid #1e2d45;border-radius:12px;padding:1.5rem;">
            <div style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:1rem;font-weight:700;margin-bottom:1rem;">Results</div>
            <div style="font-size:0.85rem;color:#8892a4;line-height:2;">
                Branch A (XGBoost+RF) → 99.70% · AUC 0.9997<br>
                Branch B (EfficientNet) → 93.27% · AUC 0.9767<br>
                Branch C (DistilBERT) → 99.87% F1 · AUC 1.0000<br>
                Fusion Layer → 99.65% · AUC 1.0000
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1rem;font-size:0.8rem;color:#8892a4;text-align:center;padding:1rem;">
        Built by <b style="color:#e8eaf0;">Avantika Bishnoi</b> ·
        M.Sc. Data Science 2024–2026 ·
        Central University of Haryana ·
        Supervisor: Dr. Keshav Singh Rawat
    </div>""", unsafe_allow_html=True)
