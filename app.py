import streamlit as st
import numpy as np
from PIL import Image

from preprocessor import preprocess
from detector import analyse, _get_deepface
from visualizer import emotion_bar_chart, race_bar_chart, annotate_face


st.set_page_config(
    page_title="FaceEQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PRE-LOAD MODELS ON STARTUP ─────────────────────────
_get_deepface()

# ── CLEAN UI ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f0f10;
    color: #e5e5e5;
}

.stApp { background-color: #0f0f10; }

.hero-title {
    font-size: 2.4rem;
    font-weight: 600;
    color: #ffffff;
}

.hero-sub {
    font-size: 0.95rem;
    color: #8a8a8a;
    margin-bottom: 1rem;
}

.result-card {
    background: #1a1a1c;
    border: 1px solid #2a2a2d;
    border-radius: 14px;
    padding: 1.5rem;
}

.attr-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #7a7a7a;
}

.attr-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    color: white;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 8px;
    font-size: 0.75rem;
    background: #2a2a2d;
    color: #d0d0d0;
    margin-right: 5px;
}

[data-testid="stFileUploadDropzone"] {
    background: #1a1a1c !important;
    border: 1px dashed #333 !important;
    border-radius: 12px !important;
}

.footer-tag {
    font-size: 0.7rem;
    color: #555;
    text-align: center;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────
st.markdown('<div class="hero-title">FaceEQ</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Facial Attribute Analysis</div>', unsafe_allow_html=True)

col_input, col_results = st.columns([1, 1.2], gap="large")

# ── INPUT ─────────────────────────────────────────────
with col_input:
    tab_upload, tab_camera = st.tabs(["Upload", "Camera"])
    image = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)

    with tab_camera:
        camera_photo = st.camera_input("Take a photo", label_visibility="collapsed")
        if camera_photo:
            image = Image.open(camera_photo).convert("RGB")

    if image is None:
        st.info("Upload or capture an image to begin analysis.")

# ── RESULTS ─────────────────────────────────────────────
with col_results:
    if image is not None:
        img_array, status = preprocess(image)

        if img_array is None:
            st.error(f"Preprocessing failed: {status}")

        else:
            with st.spinner("Analysing..."):
                try:
                    result = analyse(img_array)

                    annotated = annotate_face(
                        img_array,
                        region={},
                        label=f"{result['emotion']} · {result['age']} yrs"
                    )
                    st.image(annotated, use_container_width=True)

                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex;justify-content:space-between;">
                            <div>
                                <div class="attr-label">Age</div>
                                <div class="attr-value">{result['age']}</div>
                            </div>
                            <div>
                                <div class="attr-label">Gender</div>
                                <div class="attr-value">{result['gender'].capitalize()}</div>
                            </div>
                            <div>
                                <div class="attr-label">Time</div>
                                <div class="attr-value">{result['inference_ms']} ms</div>
                            </div>
                        </div>

                        <br>

                        <span class="badge">{result['emotion'].capitalize()}</span>
                        <span class="badge">{result['race'].capitalize()}</span>
                        <span class="badge">{result['face_count']} face</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if result['emotions']:
                        st.markdown("**Emotion Distribution**")
                        st.image(emotion_bar_chart(result['emotions']), use_container_width=True)

                    if result['race_scores']:
                        st.markdown("**Ethnicity Confidence**")
                        st.image(race_bar_chart(result['race_scores']), use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    else:
        st.markdown("""
        <div style="height:300px;display:flex;align-items:center;justify-content:center;color:#444;">
            Waiting for input...
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────
st.markdown(
    '<div class="footer-tag">FaceEQ · DeepFace Pipeline</div>',
    unsafe_allow_html=True
)