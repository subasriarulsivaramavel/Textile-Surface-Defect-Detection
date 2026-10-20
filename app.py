"""Streamlit web interface for textile surface defect detection."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.detector import detect_defects, try_yolo_predict, resize_for_display
from src.utils import bgr_to_rgb, image_to_bytes, list_images

st.set_page_config(
    page_title="Textile Defect Detection",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Light, professional theme ----------------
PRO_THEME = """
<style>
@keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.stApp {
    background: linear-gradient(135deg, #FFF5F7 0%, #FFD6A5 35%, #FDFFB6 70%, #CAFFBF 100%);
    color: #1F2937;
    overflow-x: hidden;
}
.stApp::before { content:""; position:fixed; inset:0; z-index:-1;
    background: radial-gradient(circle at 20% 30%, rgba(255,51,102,0.22) 0, transparent 38%),
        radial-gradient(circle at 80% 20%, rgba(255,159,28,0.22) 0, transparent 38%),
        radial-gradient(circle at 50% 80%, rgba(6,214,160,0.20) 0, transparent 42%),
        radial-gradient(circle at 90% 70%, rgba(255,214,10,0.20) 0, transparent 38%); animation:floaty 16s ease-in-out infinite; }
.hero-title { background: linear-gradient(90deg, #FF595E, #FF924C, #FFCA3A, #8AC926); -webkit-background-clip: text; background-clip: text; color: transparent; font-size: 2.8rem; font-weight: 800; animation:slideIn 0.8s ease-out; }
.hero-sub { color: #6B7281; font-size: 1.1rem; margin-top: 0.2rem; }
.card {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    backdrop-filter: blur(8px);
    animation: slideIn 0.5s ease-out;
    color: #1F2937;
}
.metric-card {
    background: linear-gradient(135deg, rgba(255,89,94,0.12), rgba(247,37,133,0.1));
    border: 1px solid rgba(0,0,0,0.05);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
    text-align: center;
    transition: transform 0.15s;
    color: #1F2937;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-label { color: #6B7281; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #1F2937; font-size: 1.6rem; font-weight: 700; }
.stButton > button {
    background: linear-gradient(120deg, #FF595E, #FF924C, #FFCA3A);
    color: #0F172A;
    border: none;
    border-radius: 10;
    font-weight: 700;
    padding: 0.5rem 1.2rem;
    box-shadow: 0 4px 12px rgba(255,89,94,0.3);
}
.stButton > button:hover { filter: brightness(1.05); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF, #FDF2F8);
    border-right: 1px solid rgba(0,0,0,0.08);
}
.stTabs [data-baseweb="tab"] { font-weight: 600; color: #E2E8F0; border-radius: 10px; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #FF595E, #FF924C, #8AC926);
    color: white; border-radius: 10px;
}
.img-frame img { border-radius: 12px; border: 2px solid rgba(0,0,0,0.1); box-shadow: 0 4px 14px rgba(0,0,0,0.15); }
.badge { display:inline-block; padding:0.25rem 0.5rem; border-radius:999px; font-weight:700; font-size:0.75rem; margin:0.1rem 0; }
.badge-low { background:#22C55E33; color:#15803D; border:1px solid #22C55E; }
.badge-medium { background:#F59E0B33; color:#B45309; border:1px solid #F59E0B; }
.badge-high { background:#EF444433; color:#B91C1C; border:1px solid #EF4444; }
.badge-none { background:#64748B33; color:#374151; border:1px solid #64748B; }
.gallery img { border-radius:8px; cursor:pointer; transition:0.2s; }
.gallery img:hover { transform: scale(1.03); box-shadow:0 0 12px #FF595E; }
hr { border-color: rgba(0,0,0,0.1); }
</style>
"""
st.markdown(PRO_THEME, unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">🧵 Textile Surface Defect Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Professional computer-vision inspection for fabric quality.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ Settings")
    sensitivity = st.slider("Sensitivity", 0.3, 3.0, 1.2, 0.1, key="sens")
    min_area = st.slider("Min. defect area (px)", 20, 500, 80, 10, key="ma")
    blur_kernel = st.select_slider("Blur kernel", options=[3, 5, 7, 9], value=5, key="bk")
    bg_kernel = st.select_slider("Background kernel", options=[31, 51, 71], value=31, key="bgk")
    st.divider()
    st.markdown("### 📚 Dataset")
    st.markdown("[Mendeley Dataset](https://data.mendeley.com/datasets/6574nhzm8x/1)")
    st.markdown("[Reference Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11648198/)")

tabs = st.tabs(["🖼️ Single image", "📁 Batch", "📸 Gallery", "ℹ️ About"])

# ---------- Helpers ----------
def run_detect(img):
    return detect_defects(img, sensitivity=sensitivity, min_area=min_area, blur_kernel=blur_kernel, bg_kernel=bg_kernel)

def render_result(res, boxes):
    sev = res["metrics"]["severity"]
    badge = {"Low":"badge-low","Medium":"badge-medium","High":"badge-high","None":"badge-none"}[sev]
    st.markdown(f'<span class="badge {badge}">Severity: {sev}</span>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,label,val in [(c1,"Defects",res["metrics"]["num_defects"]),(c2,"Area %",f"{res['metrics']['defect_percent']}%"),
                          (c3,"Severity",sev),(c4,"Size",res["metrics"]["image_size"])]:
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)
    st.subheader("Results")
    r1,r2 = st.columns(2)
    with r1:
        st.image(bgr_to_rgb(resize_for_display(res["original"])), caption="Original", use_container_width=True)
        st.image(resize_for_display(res["diff"]), caption="Deviation", use_container_width=True)
    with r2:
        st.image(resize_for_display(res["mask"]), caption="Defect mask", use_container_width=True)
        st.image(bgr_to_rgb(resize_for_display(res["annotated"])), caption="Detected defects", use_container_width=True)
    if boxes:
        st.subheader("Defect regions")
        st.dataframe(boxes, width="stretch")
    buf = image_to_bytes(res["annotated"])
    st.download_button("⬇️ Download annotated", data=buf, file_name="defect_annotated.png", mime="image/png")

# ---------- Single image ----------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    src = st.radio("Source", ["Upload","Sample"], horizontal=True, key="src")
    img_bgr = None
    if src == "Upload":
        up = st.file_uploader("Image", ["png","jpg","jpeg","bmp","tif","tiff"], key="uf")
        if up:
            arr = np.frombuffer(up.read(), np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        sam = list_images(str(ROOT/"samples"))
        if not sam: st.warning("Run `python generate_samples.py`.")
        else:
            sel = st.selectbox("Sample", sam, index=0, key="sel", format_func=lambda p: Path(p).name)
            img_bgr = cv2.imread(sel)
    go = st.button("🔍 Detect", use_container_width=True, disabled=(img_bgr is None))
    st.markdown("</div>", unsafe_allow_html=True)
    if img_bgr is not None and go:
        with st.spinner("Analyzing..."):
            res = run_detect(img_bgr)
        render_result(res, res["boxes"])

# ---------- Batch ----------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Batch process a folder.")
    folder = st.text_input("Folder", str(ROOT/"samples"), key="bf")
    if st.button("Run batch", use_container_width=True):
        import pandas as pd
        from src.utils import list_images
        paths = list_images(folder)
        if not paths: st.error("No images found.")
        else:
            rows = []
            for p in paths:
                I = cv2.imread(p)
                if I is None: continue
                r = run_detect(I)
                rows.append({"file":p.split("\\")[-1], **r["metrics"]})
            st.dataframe(pd.DataFrame(rows), width="stretch")
            csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "report.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Gallery ----------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Samples")
    sam = list_images(str(ROOT/"samples"))
    if not sam: st.warning("No samples.")
    else:
        for i,p in enumerate(sam):
            with st.columns(3)[i%3]:
                st.image(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB), p.split("\\")[-1], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- About ----------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Classical OpenCV pipeline for fabric defect detection. "
             "Benchmark against YOLO/CNN models. Runs fully on CPU.")
    st.markdown("```\npython -m streamlit run app.py --server.headless true\n```")
    st.markdown("<div style='text-align:center; color:#6B7281;'>Made for textile quality inspection</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)