import streamlit as st
import numpy as np
import json
import time
import base64
import io
from pathlib import Path
from PIL import Image

# Page config
st.set_page_config(
    page_title="X-Ray Threat Detector",
    page_icon="https://cdn-icons-png.flaticon.com/512/2991/2991108.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths  (works both locally and on Streamlit Cloud)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_model_part2.tflite"
CLASS_NAMES_PATH = PROJECT_ROOT / "saved_models" / "class_names_7classes.npy"
BG_IMAGE_PATH = PROJECT_ROOT / "3rdairport.jpg"

# Threat metadata  (clean labels, risk levels, descriptions)
THREAT_INFO = {
    "Battery": {
        "risk": "Medium",
        "color": "#f0ad4e",
        "icon": "https://cdn-icons-png.flaticon.com/512/3659/3659898.png",
        "desc": "Lithium batteries can pose fire risks in cargo.",
    },
    "Scissors": {
        "risk": "Medium",
        "color": "#f0ad4e",
        "icon": "https://cdn-icons-png.flaticon.com/512/2544/2544tried.png",
        "desc": "Scissors with blades over 6 cm are restricted.",
    },
    "Hammer": {
        "risk": "Medium",
        "color": "#f0ad4e",
        "icon": "https://cdn-icons-png.flaticon.com/512/595/595584.png",
        "desc": "Blunt-force tools are prohibited in carry-on luggage.",
    },
    "Pliers": {
        "risk": "Medium",
        "color": "#f0ad4e",
        "icon": "https://cdn-icons-png.flaticon.com/512/595/595584.png",
        "desc": "Hand tools over 7 cm are restricted in cabins.",
    },
    "Wrench": {
        "risk": "Medium",
        "color": "#f0ad4e",
        "icon": "https://cdn-icons-png.flaticon.com/512/595/595584.png",
        "desc": "Heavy tools are not allowed as carry-on items.",
    },
    "Explosive": {
        "risk": "Critical",
        "color": "#d9534f",
        "icon": "https://cdn-icons-png.flaticon.com/512/2785/2785819.png",
        "desc": "Explosive materials are strictly prohibited.",
    },
    "Bullet": {
        "risk": "High",
        "color": "#d9534f",
        "icon": "https://cdn-icons-png.flaticon.com/512/2785/2785819.png",
        "desc": "Ammunition is banned from all passenger flights.",
    },
    "Knife": {
        "risk": "High",
        "color": "#d9534f",
        "icon": "https://cdn-icons-png.flaticon.com/512/3460/3460996.png",
        "desc": "Knives of any length are prohibited in carry-on.",
    },
    "Cutter": {
        "risk": "High",
        "color": "#d9534f",
        "icon": "https://cdn-icons-png.flaticon.com/512/3460/3460996.png",
        "desc": "Box cutters and blades are strictly forbidden.",
    },
    "Lighter": {
        "risk": "Low",
        "color": "#5bc0de",
        "icon": "https://cdn-icons-png.flaticon.com/512/785/785116.png",
        "desc": "One disposable lighter is allowed on person only.",
    },
}


def clean_class_name(raw_name):
    """Turn 'Class 10_Battery' into 'Battery'."""
    parts = raw_name.split("_", 1)
    if len(parts) == 2:
        return parts[1]
    return raw_name


# Load model and class names  (cached so it only runs once)
@st.cache_resource
def load_model():
    """Load the TFLite model and return the interpreter."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


@st.cache_data
def load_class_names():
    """Load class names from .npy and return cleaned list."""
    raw_names = np.load(str(CLASS_NAMES_PATH), allow_pickle=True)
    return [clean_class_name(str(n)) for n in raw_names]


def predict(interpreter, image):
    """Run inference on a PIL image and return (class_name, confidence, all_probs)."""
    class_names = load_class_names()

    # Get model input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]  # e.g. [1, 224, 224, 3]
    height, width = input_shape[1], input_shape[2]

    # Preprocess: resize, convert to float32, apply EfficientNet preprocessing
    img = image.convert("RGB").resize((width, height))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # EfficientNetB0 expects pixels preprocessed (scales to [-1, 1])
    import tensorflow as tf
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    probs = output[0]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    predicted_class = class_names[top_idx]

    return predicted_class, confidence, probs


# Background image helper
def get_bg_css():
    """Return CSS string with background image + subtle shader blobs."""
    if BG_IMAGE_PATH.exists():
        with open(BG_IMAGE_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode()

        return f"""
        <style>
        /* Full-page background */
        .stApp {{
            background-image:
              linear-gradient(180deg, rgba(2,6,23,0.55), rgba(2,6,23,0.70)),
              url("data:image/jpeg;base64,{data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Soft "glass" wash so text stays readable */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(15, 23, 42, 0.35);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            z-index: -1;
        }}

        /* Shader blobs (subtle glow) */
        .stApp::after {{
            content: "";
            position: fixed;
            inset: -20%;
            background:
              radial-gradient(600px 420px at 18% 22%, rgba(56,189,248,0.18), transparent 55%),
              radial-gradient(520px 380px at 78% 28%, rgba(129,140,248,0.14), transparent 55%),
              radial-gradient(680px 520px at 55% 78%, rgba(34,197,94,0.10), transparent 60%);
            filter: blur(18px);
            opacity: 0.9;
            z-index: -2;
            pointer-events: none;
            animation: shaderDrift 14s ease-in-out infinite;
        }}

        @keyframes shaderDrift {{
          0%   {{ transform: translate3d(0,0,0) scale(1); }}
          50%  {{ transform: translate3d(-2%, 1%, 0) scale(1.02); }}
          100% {{ transform: translate3d(0,0,0) scale(1); }}
        }}
        </style>
        """
    return ""


# Custom CSS for the entire app
CUSTOM_CSS = """
<style>
/* Modern font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root{
  --text: #e5e7eb;
  --muted: rgba(229,231,235,0.72);
  --card: rgba(15, 23, 42, 0.62);
  --card2: rgba(15, 23, 42, 0.78);
  --border: rgba(148,163,184,0.18);
  --accent: #38bdf8;
}

* { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

/* Remove Streamlit top padding + default chrome */
div[data-testid="stAppViewContainer"] { padding-top: 0rem; }
header[data-testid="stHeader"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Make content feel centered and premium */
.block-container {
  padding-top: 1.8rem;
  padding-bottom: 3rem;
  max-width: 1200px;
}

/* Tab bar styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.70);
    border-radius: 12px;
    padding: 0.3rem;
    gap: 0;
    width: 100%;
}
.stTabs [data-baseweb="tab"] {
    flex: 1;
    justify-content: center;
    color: #94a3b8;
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.8rem 1.5rem;
    border-radius: 10px;
    background: transparent;
    border: none;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0;
    background: rgba(56, 189, 248, 0.15);
}
.stTabs [aria-selected="true"] {
    background: rgba(56, 189, 248, 0.25) !important;
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #38bdf8 !important;
    height: 3px;
    border-radius: 3px;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* Header styling */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    animation: fadeInDown 0.8s ease-out;
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.main-header p {
    color: #ffffff;
    font-size: 1.1rem;
    font-weight: 300;
}

/* Upload area */
.upload-area {
    border: 2px dashed #475569;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(10px);
}
.upload-area:hover {
    border-color: #38bdf8;
    background: rgba(30, 41, 59, 0.8);
}

/* Result card */
.result-card {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    animation: floatIn 650ms ease-out both;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.risk-critical { background: rgba(239,68,68,0.2); color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); }
.risk-high { background: rgba(249,115,22,0.2); color: #fdba74; border: 1px solid rgba(249,115,22,0.4); }
.risk-medium { background: rgba(234,179,8,0.2); color: #fde047; border: 1px solid rgba(234,179,8,0.4); }
.risk-low { background: rgba(34,197,94,0.2); color: #86efac; border: 1px solid rgba(34,197,94,0.4); }

/* Confidence meter */
.conf-bar-bg {
    width: 100%;
    height: 12px;
    background: rgba(51,65,85,0.8);
    border-radius: 6px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 1.2s ease-out;
}

/* Probability table */
.prob-row {
    display: flex;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(71,85,105,0.3);
}
.prob-label {
    flex: 1;
    color: rgba(226,232,240,0.92);
    font-size: 0.9rem;
}
.prob-value {
    color: rgba(226,232,240,0.92);
    font-weight: 600;
    font-size: 0.9rem;
    min-width: 50px;
    text-align: right;
}
.prob-bar-bg {
    flex: 2;
    height: 6px;
    background: rgba(51,65,85,0.6);
    border-radius: 3px;
    margin: 0 0.8rem;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    border-radius: 3px;
}

/* Carousel */
.carousel-outer {
    overflow: hidden;
    width: 100%;
    padding: 1rem 0;
}
.carousel-track {
    display: flex;
    gap: 1.2rem;
    width: max-content;
    animation: scrollCarousel 25s linear infinite;
}
.carousel-outer:hover .carousel-track {
    animation-play-state: paused;
}
.carousel-card {
    min-width: 240px;
    background: rgba(15, 23, 42, 0.92);
    border-radius: 12px;
    padding: 1.4rem;
    border: 1px solid rgba(71,85,105,0.5);
    text-align: center;
    transition: transform 0.3s ease, border-color 0.3s ease;
    flex-shrink: 0;
    transform: perspective(900px) translateY(0) rotateX(0);
}
.carousel-card:hover {
    transform: perspective(900px) translateY(-6px) rotateX(2deg);
    border-color: #38bdf8;
}
.carousel-card h4 {
    color: #e2e8f0;
    margin: 0.5rem 0 0.3rem 0;
    font-size: 1.2rem;
}
.carousel-card p {
    color: #c0c8d4;
    font-size: 0.95rem;
    margin: 0;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(10px);
}

/* Animations */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes floatIn {
    from { opacity: 0; transform: translateY(18px) scale(0.985); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
@keyframes scrollCarousel {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* Typewriter effect */
.typewrite {
  display: inline-block;
  overflow: hidden;
  white-space: nowrap;
  border-right: 2px solid rgba(255,255,255,0.65);
  animation: typing 1.6s steps(18, end), caret 0.9s step-end infinite;
}

@keyframes typing {
  from { width: 0; }
  to   { width: 100%; }
}

@keyframes caret {
  50% { border-color: transparent; }
}

/* Spinner */
.scan-animation {
    text-align: center;
    padding: 2rem;
    animation: pulse 1.5s ease-in-out infinite;
}
.scan-animation p {
    color: #38bdf8;
    font-size: 1.1rem;
    font-weight: 500;
}

/* Scan line overlay */
.scan-box {
  position: relative;
  border-radius: 18px;
  border: 1px solid rgba(56,189,248,0.22);
  background: rgba(2,6,23,0.45);
  overflow: hidden;
  padding: 2.2rem 1.2rem;
}

.scan-line{
  position:absolute;
  left:0; right:0;
  height: 3px;
  background: rgba(56,189,248,0.75);
  filter: blur(0.2px);
  animation: scanMove 1.1s linear infinite;
}

@keyframes scanMove{
  0% { top: -10%; opacity: 0.0; }
  10%{ opacity: 1.0; }
  90%{ opacity: 1.0; }
  100%{ top: 110%; opacity: 0.0; }
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    color: #64748b;
    font-size: 0.8rem;
    border-top: 1px solid rgba(71,85,105,0.3);
    margin-top: 3rem;
}

/* Hero section */
.hero{
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(2,6,23,0.35), rgba(2,6,23,0.25));
  border-radius: 24px;
  padding: 3.2rem 2.4rem;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow: 0 18px 55px rgba(0,0,0,0.35);
  margin-bottom: 1.8rem;
  animation: floatIn 700ms ease-out both;
  position: relative;
  overflow: hidden;
}

/* Star layers */
.hero::before,
.hero::after{
  content:"";
  position:absolute;
  inset:-30%;
  pointer-events:none;
  z-index: 0;
  opacity: 0.55;
  mix-blend-mode: screen;
  filter: blur(0.2px);
}

/* Small stars layer */
.hero::before{
  background:
    radial-gradient(1px 1px at 12% 18%, rgba(255,255,255,0.85), transparent 60%),
    radial-gradient(1px 1px at 28% 36%, rgba(255,255,255,0.75), transparent 60%),
    radial-gradient(1px 1px at 44% 22%, rgba(255,255,255,0.65), transparent 60%),
    radial-gradient(1px 1px at 66% 28%, rgba(255,255,255,0.80), transparent 60%),
    radial-gradient(1px 1px at 78% 42%, rgba(255,255,255,0.70), transparent 60%),
    radial-gradient(1px 1px at 86% 18%, rgba(255,255,255,0.60), transparent 60%),
    radial-gradient(1px 1px at 18% 62%, rgba(255,255,255,0.70), transparent 60%),
    radial-gradient(1px 1px at 40% 70%, rgba(255,255,255,0.60), transparent 60%),
    radial-gradient(1px 1px at 58% 76%, rgba(255,255,255,0.78), transparent 60%),
    radial-gradient(1px 1px at 74% 64%, rgba(255,255,255,0.62), transparent 60%),
    radial-gradient(1px 1px at 90% 74%, rgba(255,255,255,0.72), transparent 60%);
  animation: twinkle1 3.8s ease-in-out infinite;
}

/* Bigger sparkle stars layer */
.hero::after{
  background:
    radial-gradient(2px 2px at 22% 26%, rgba(255,255,255,0.75), transparent 65%),
    radial-gradient(2px 2px at 52% 16%, rgba(255,255,255,0.65), transparent 65%),
    radial-gradient(2px 2px at 72% 24%, rgba(255,255,255,0.70), transparent 65%),
    radial-gradient(2px 2px at 84% 54%, rgba(255,255,255,0.60), transparent 65%),
    radial-gradient(2px 2px at 36% 56%, rgba(255,255,255,0.62), transparent 65%),
    radial-gradient(2px 2px at 62% 74%, rgba(255,255,255,0.68), transparent 65%);
  opacity: 0.35;
  animation: twinkle2 5.6s ease-in-out infinite;
}

/* Bright spark stars (diamond shaped) */
.hero .spark{
  position:absolute;
  width: 4px;
  height: 4px;
  background: rgba(255,255,255,0.9);
  box-shadow: 0 0 8px rgba(255,255,255,0.5);
  opacity: 0;
  pointer-events:none;
  z-index: 0;
  transform: rotate(45deg);
}

.spark-1{ top:12%; left:8%;  animation: sparkle 3.2s 0.0s ease-in-out infinite; }
.spark-2{ top:22%; left:25%; animation: sparkle 4.0s 0.6s ease-in-out infinite; }
.spark-3{ top:8%;  left:42%; animation: sparkle 3.6s 1.2s ease-in-out infinite; }
.spark-4{ top:18%; left:60%; animation: sparkle 4.4s 0.3s ease-in-out infinite; }
.spark-5{ top:10%; left:78%; animation: sparkle 3.8s 1.8s ease-in-out infinite; }
.spark-6{ top:28%; left:90%; animation: sparkle 4.2s 0.9s ease-in-out infinite; }
.spark-7{ top:40%; left:15%; animation: sparkle 3.4s 2.1s ease-in-out infinite; }
.spark-8{ top:55%; left:35%; animation: sparkle 4.6s 1.5s ease-in-out infinite; }
.spark-9{ top:48%; left:55%; animation: sparkle 3.9s 0.4s ease-in-out infinite; }
.spark-10{ top:42%; left:72%; animation: sparkle 4.1s 2.4s ease-in-out infinite; }
.spark-11{ top:60%; left:88%; animation: sparkle 3.5s 1.0s ease-in-out infinite; }
.spark-12{ top:65%; left:20%; animation: sparkle 4.3s 1.7s ease-in-out infinite; }
.spark-13{ top:72%; left:45%; animation: sparkle 3.7s 2.8s ease-in-out infinite; }
.spark-14{ top:68%; left:65%; animation: sparkle 4.5s 0.7s ease-in-out infinite; }
.spark-15{ top:78%; left:82%; animation: sparkle 3.3s 2.0s ease-in-out infinite; }
.spark-16{ top:35%; left:48%; animation: sparkle 4.8s 1.3s ease-in-out infinite; }
.spark-17{ top:15%; left:52%; animation: sparkle 3.1s 2.6s ease-in-out infinite; }
.spark-18{ top:82%; left:10%; animation: sparkle 4.0s 0.2s ease-in-out infinite; }
.spark-19{ top:50%; left:5%;  animation: sparkle 3.6s 1.9s ease-in-out infinite; }
.spark-20{ top:85%; left:58%; animation: sparkle 4.7s 2.3s ease-in-out infinite; }

@keyframes twinkle1{
  0%, 100% { opacity: 0.35; transform: translate3d(0,0,0); }
  50%      { opacity: 0.70; transform: translate3d(-0.6%, 0.4%, 0); }
}

@keyframes twinkle2{
  0%, 100% { opacity: 0.20; transform: translate3d(0,0,0) scale(1); }
  50%      { opacity: 0.55; transform: translate3d(0.7%, -0.4%, 0) scale(1.01); }
}

@keyframes sparkle{
  0%, 65%, 100% { opacity: 0; transform: scale(0.6); }
  72%           { opacity: 1; transform: scale(1.0); }
  80%           { opacity: 0.2; transform: scale(1.6); }
}

.hero-inner{
  text-align: center;
  max-width: 820px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}

.hero-kicker{
  color: rgba(255,255,255,0.75);
  font-size: 1.05rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}

.hero-title{
  color: #ffffff;
  font-size: 4rem;
  line-height: 1.0;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin-bottom: 0.8rem;
  text-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

.hero-sub{
  color: rgba(255,255,255,0.78);
  font-size: 1.08rem;
  line-height: 1.6;
  margin: 0 auto 1.8rem auto;
}

.hero-actions{
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1.2rem;
}

.hero-btn{
  border-radius: 999px;
  padding: 0.85rem 1.6rem;
  font-weight: 600;
  font-size: 1rem;
  border: 1px solid var(--border);
  user-select: none;
  position: relative;
  overflow: hidden;
  transition: transform 180ms ease, box-shadow 180ms ease;
}

.hero-btn:hover{
  transform: translateY(-2px);
  box-shadow: 0 16px 40px rgba(0,0,0,0.25);
}

.hero-btn::after{
  content:"";
  position:absolute;
  inset:-40%;
  background: linear-gradient(120deg, transparent, rgba(255,255,255,0.25), transparent);
  transform: translateX(-60%);
  transition: transform 450ms ease;
}

.hero-btn:hover::after{
  transform: translateX(60%);
}

.hero-btn.primary{
  background: rgba(255,255,255,0.92);
  color: rgba(2,6,23,0.95);
}

.hero-btn.ghost{
  background: rgba(255,255,255,0.10);
  color: rgba(255,255,255,0.85);
}

.hero-pill{
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.55rem 1rem;
  border-radius: 999px;
  border: 1px solid rgba(34,197,94,0.25);
  background: rgba(34,197,94,0.10);
  color: rgba(255,255,255,0.80);
  font-size: 0.95rem;
  position: relative;
  overflow: hidden;
}

.hero-pill::after{
  content:"";
  position:absolute;
  inset:-60%;
  background: radial-gradient(circle, rgba(34,197,94,0.20), transparent 55%);
  animation: glowPulse 2.2s ease-in-out infinite;
}

@keyframes glowPulse{
  0%,100%{ transform: scale(0.95); opacity: 0.35; }
  50%    { transform: scale(1.05); opacity: 0.70; }
}

.hero-pill .dot{
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: rgba(34,197,94,0.95);
  box-shadow: 0 0 14px rgba(34,197,94,0.6);
}

.hero-hint{
  margin-top: 1.8rem;
  color: rgba(255,255,255,0.55);
  font-size: 0.9rem;
}

.hero-hint .arrow{
  margin-top: 0.35rem;
  font-size: 1.1rem;
  opacity: 0.9;
}
</style>
"""


# Carousel of detectable threats
def render_carousel():
    """Render a horizontal scrollable carousel of threat types."""
    cards_html = ""
    for name, info in THREAT_INFO.items():
        risk_class = "risk-" + info["risk"].lower()
        cards_html += (
            '<div class="carousel-card">'
            '<span class="risk-badge {risk_class}">{risk}</span>'
            '<h4>{name}</h4>'
            '<p>{desc}</p>'
            '</div>'
        ).format(
            risk_class=risk_class,
            risk=info["risk"],
            name=name,
            desc=info["desc"],
        )

    # Duplicate cards so the scroll loops seamlessly
    st.markdown(
        '<div class="carousel-outer"><div class="carousel-track">'
        '{cards}{cards}</div></div>'.format(cards=cards_html),
        unsafe_allow_html=True,
    )


# Render prediction results
def render_results(predicted_class, confidence, probs):
    """Display the prediction results in a styled card layout."""
    class_names = load_class_names()
    info = THREAT_INFO.get(predicted_class, {
        "risk": "Unknown",
        "color": "#94a3b8",
        "desc": "No additional information available.",
    })

    risk = info["risk"]
    risk_class = "risk-" + risk.lower()

    # Low-confidence override
    is_uncertain = confidence < 0.30
    if is_uncertain:
        display_name = "Not Known"
        display_risk = "Unknown"
        risk_class = "risk-low"
        display_desc = "The model is not confident enough. Closest guess: " + predicted_class + "."
    else:
        display_name = predicted_class
        display_risk = risk
        display_desc = info["desc"]

    # Confidence color
    if confidence >= 0.8:
        conf_color = "#22c55e"
    elif confidence >= 0.5:
        conf_color = "#eab308"
    else:
        conf_color = "#ef4444"

    conf_pct = str(round(confidence * 100, 1))
    conf_width = str(round(confidence * 100))

    # Top result card
    st.markdown("""
    <div class="result-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
            <div>
                <p style="color:#94a3b8; font-size:0.85rem; margin:0;">DETECTED OBJECT</p>
                <h2 style="color:#e2e8f0; margin:0.2rem 0; font-size:2rem;">{name}</h2>
                <span class="risk-badge {risk_class}">{risk} Risk</span>
            </div>
            <div style="text-align:right;">
                <p style="color:#94a3b8; font-size:0.85rem; margin:0;">CONFIDENCE</p>
                <p style="color:{conf_color}; font-size:2.2rem; font-weight:700; margin:0;">{conf_pct}%</p>
            </div>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{conf_width}%; background:{conf_color};"></div>
        </div>
        <p style="color:#94a3b8; margin-top:1rem; font-size:0.95rem;">{desc}</p>
    </div>
    """.format(
        name=display_name,
        risk_class=risk_class,
        risk=display_risk,
        conf_color=conf_color,
        conf_pct=conf_pct,
        conf_width=conf_width,
        desc=display_desc,
    ), unsafe_allow_html=True)

    # Probability breakdown
    st.markdown(
        "<h3 style='color:#ffffff; margin-top:2rem;'>All Class Probabilities</h3>",
        unsafe_allow_html=True,
    )

    # Sort by probability descending
    sorted_indices = np.argsort(probs)[::-1]

    rows_html = ""
    for idx in sorted_indices:
        name = class_names[idx]
        prob = float(probs[idx])
        pct = str(round(prob * 100, 1))
        bar_w = str(round(prob * 100))
        rows_html += (
            '<div class="prob-row">'
            '<span class="prob-label">{name}</span>'
            '<div class="prob-bar-bg">'
            '<div class="prob-bar-fill" style="width:{bar_w}%;"></div>'
            '</div>'
            '<span class="prob-value">{pct}%</span>'
            '</div>'
        ).format(name=name, bar_w=bar_w, pct=pct)

    st.markdown(
        '<div class="result-card" style="margin-top:1rem;">{rows}</div>'.format(
            rows=rows_html
        ),
        unsafe_allow_html=True,
    )


# Sidebar
def render_sidebar():
    """Render the sidebar with app info and settings."""
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#e2e8f0;'>About</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.9rem;'>"
            "This application uses a deep learning model trained on the "
            "STCray X-ray dataset to detect threats in baggage scans. "
            "The model is an EfficientNetB0 fine-tuned on 7 threat classes."
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h3 style='color:#e2e8f0; margin-top:1.5rem;'>Model Details</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div style="background:rgba(30,41,59,0.6); border-radius:12px; padding:1rem; border:1px solid rgba(71,85,105,0.4);">
            <p style="color:#94a3b8; margin:0.3rem 0; font-size:0.85rem;">
                <strong style="color:#e2e8f0;">Architecture:</strong> EfficientNetB0</p>
            <p style="color:#94a3b8; margin:0.3rem 0; font-size:0.85rem;">
                <strong style="color:#e2e8f0;">Format:</strong> TensorFlow Lite</p>
            <p style="color:#94a3b8; margin:0.3rem 0; font-size:0.85rem;">
                <strong style="color:#e2e8f0;">Classes:</strong> 7 threat types</p>
            <p style="color:#94a3b8; margin:0.3rem 0; font-size:0.85rem;">
                <strong style="color:#e2e8f0;">Input Size:</strong> 224 x 224 px</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<h3 style='color:#e2e8f0; margin-top:1.5rem;'>How to Use</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div style="color:#94a3b8; font-size:0.85rem; line-height:1.6;">
            <p>1. Upload an X-ray scan image using the uploader.</p>
            <p>2. The model will automatically analyse the image.</p>
            <p>3. View the predicted threat class, confidence score, and probability breakdown.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div class='footer'>DLOR Project -- Temasek Polytechnic</div>",
            unsafe_allow_html=True,
        )


# Image converter
FORMAT_OPTIONS = {
    "PNG": {"ext": "png", "mime": "image/png", "pillow": "PNG"},
    "JPEG": {"ext": "jpg", "mime": "image/jpeg", "pillow": "JPEG"},
    "BMP": {"ext": "bmp", "mime": "image/bmp", "pillow": "BMP"},
    "GIF": {"ext": "gif", "mime": "image/gif", "pillow": "GIF"},
    "TIFF": {"ext": "tiff", "mime": "image/tiff", "pillow": "TIFF"},
    "WEBP": {"ext": "webp", "mime": "image/webp", "pillow": "WEBP"},
    "ICO": {"ext": "ico", "mime": "image/x-icon", "pillow": "ICO"},
}


def render_image_converter():
    """Render the image format converter tab."""
    st.markdown(
        "<h3 style='color:#ffffff; font-size:2.5rem;'>Image Format Converter</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#ffffff; font-size:1.9rem;'>"
        "Upload any image and convert it to a different format. "
        "Supports PNG, JPEG, BMP, GIF, TIFF, WEBP, and ICO.</p>",
        unsafe_allow_html=True,
    )

    col_upload, col_output = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown(
            "<h4 style='color:#ffffff;'>1. Upload Image</h4>",
            unsafe_allow_html=True,
        )
        conv_file = st.file_uploader(
            "Choose an image to convert",
            type=["png", "jpg", "jpeg", "bmp", "gif", "tiff", "tif", "webp", "ico"],
            label_visibility="collapsed",
            key="converter_uploader",
        )

        if conv_file is not None:
            conv_image = Image.open(conv_file)
            st.image(conv_image, caption="Original image", use_container_width=True)

            # Show original info
            orig_format = conv_image.format if conv_image.format else "Unknown"
            orig_size = conv_image.size
            st.markdown(
                '<div class="result-card">'
                '<p style="color:#ffffff; margin:0.2rem 0; font-size:0.9rem;">'
                '<strong>Original format:</strong> {fmt}</p>'
                '<p style="color:#ffffff; margin:0.2rem 0; font-size:0.9rem;">'
                '<strong>Dimensions:</strong> {w} x {h} px</p>'
                '<p style="color:#ffffff; margin:0.2rem 0; font-size:0.9rem;">'
                '<strong>Mode:</strong> {mode}</p>'
                '</div>'.format(
                    fmt=orig_format,
                    w=orig_size[0],
                    h=orig_size[1],
                    mode=conv_image.mode,
                ),
                unsafe_allow_html=True,
            )

    with col_output:
        st.markdown(
            "<h4 style='color:#ffffff;'>2. Choose Output Format</h4>",
            unsafe_allow_html=True,
        )

        if conv_file is not None:
            target_format = st.selectbox(
                "Convert to",
                list(FORMAT_OPTIONS.keys()),
                label_visibility="collapsed",
            )

            # Quality slider for JPEG / WEBP
            quality = 95
            if target_format in ("JPEG", "WEBP"):
                quality = st.slider("Quality", 10, 100, 95)

            # Resize option
            resize = st.checkbox("Resize image")
            new_w, new_h = conv_image.size
            if resize:
                rc1, rc2 = st.columns(2)
                with rc1:
                    new_w = st.number_input("Width (px)", min_value=1, value=conv_image.size[0])
                with rc2:
                    new_h = st.number_input("Height (px)", min_value=1, value=conv_image.size[1])

            if st.button("Convert", type="primary", use_container_width=True):
                fmt_info = FORMAT_OPTIONS[target_format]
                out_image = conv_image.copy()

                # Resize if requested
                if resize:
                    out_image = out_image.resize((int(new_w), int(new_h)), Image.LANCZOS)

                # Convert RGBA to RGB for formats that don't support alpha
                if target_format in ("JPEG", "BMP", "ICO") and out_image.mode == "RGBA":
                    bg = Image.new("RGB", out_image.size, (255, 255, 255))
                    bg.paste(out_image, mask=out_image.split()[3])
                    out_image = bg
                elif target_format == "JPEG" and out_image.mode != "RGB":
                    out_image = out_image.convert("RGB")

                # ICO size limit
                if target_format == "ICO":
                    max_dim = max(out_image.size)
                    if max_dim > 256:
                        ratio = 256.0 / max_dim
                        out_image = out_image.resize(
                            (int(out_image.size[0] * ratio), int(out_image.size[1] * ratio)),
                            Image.LANCZOS,
                        )

                # Save to buffer
                buf = io.BytesIO()
                save_kwargs = {"format": fmt_info["pillow"]}
                if target_format in ("JPEG", "WEBP"):
                    save_kwargs["quality"] = quality
                out_image.save(buf, **save_kwargs)
                buf.seek(0)

                file_size = len(buf.getvalue())
                if file_size < 1024:
                    size_str = str(file_size) + " B"
                elif file_size < 1024 * 1024:
                    size_str = str(round(file_size / 1024, 1)) + " KB"
                else:
                    size_str = str(round(file_size / (1024 * 1024), 2)) + " MB"

                st.markdown(
                    '<div class="result-card">'
                    '<p style="color:#22c55e; font-size:1.1rem; font-weight:600; margin:0;">'
                    'Conversion successful</p>'
                    '<p style="color:#ffffff; margin:0.3rem 0; font-size:0.9rem;">'
                    '<strong>Output format:</strong> {fmt}</p>'
                    '<p style="color:#ffffff; margin:0.3rem 0; font-size:0.9rem;">'
                    '<strong>Output size:</strong> {w} x {h} px</p>'
                    '<p style="color:#ffffff; margin:0.3rem 0; font-size:0.9rem;">'
                    '<strong>File size:</strong> {fsize}</p>'
                    '</div>'.format(
                        fmt=target_format,
                        w=out_image.size[0],
                        h=out_image.size[1],
                        fsize=size_str,
                    ),
                    unsafe_allow_html=True,
                )

                # Preview
                st.image(out_image, caption="Converted image", use_container_width=True)

                # Filename
                orig_stem = Path(conv_file.name).stem
                out_name = orig_stem + "_converted." + fmt_info["ext"]

                st.download_button(
                    label="Download " + target_format,
                    data=buf,
                    file_name=out_name,
                    mime=fmt_info["mime"],
                    use_container_width=True,
                )
        else:
            st.markdown("""
            <div class="upload-area">
                <p style="color:#ffffff; font-size:1.1rem; margin:0;">
                    Upload an image on the left to get started
                </p>
                <p style="color:#e2e8f0; font-size:0.85rem; margin-top:0.5rem;">
                    Supports PNG, JPG, BMP, GIF, TIFF, WEBP, ICO
                </p>
            </div>
            """, unsafe_allow_html=True)


# Main app
def main():
    """Main application entry point."""
    # Inject styles
    st.markdown(get_bg_css(), unsafe_allow_html=True)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Header (hero section)
    st.markdown(
        '<div class="hero">'
        '<div class="spark spark-1"></div>'
        '<div class="spark spark-2"></div>'
        '<div class="spark spark-3"></div>'
        '<div class="spark spark-4"></div>'
        '<div class="spark spark-5"></div>'
        '<div class="spark spark-6"></div>'
        '<div class="spark spark-7"></div>'
        '<div class="spark spark-8"></div>'
        '<div class="spark spark-9"></div>'
        '<div class="spark spark-10"></div>'
        '<div class="spark spark-11"></div>'
        '<div class="spark spark-12"></div>'
        '<div class="spark spark-13"></div>'
        '<div class="spark spark-14"></div>'
        '<div class="spark spark-15"></div>'
        '<div class="spark spark-16"></div>'
        '<div class="spark spark-17"></div>'
        '<div class="spark spark-18"></div>'
        '<div class="spark spark-19"></div>'
        '<div class="spark spark-20"></div>'
        '<div class="hero-inner">'
        '<div class="hero-kicker">X-Ray Threat</div>'
        '<div class="hero-title"><span class="typewrite">Detector</span></div>'
        '<div class="hero-sub">'
        'AI-powered baggage screening for aviation security. '
        'Upload an X-ray scan and get instant threat classification.'
        '</div>'
        '<div class="hero-actions">'
        '<div class="hero-btn primary">Start Scanning</div>'
        '<div class="hero-btn ghost">View All Classes</div>'
        '</div>'
        '<div class="hero-pill">'
        '<span class="dot"></span>'
        '<span>Model loaded &amp; ready</span>'
        '</div>'
        '<div class="hero-hint">'
        'Scroll to explore'
        '<div class="arrow">&#8595;</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab_scan, tab_convert = st.tabs(["Threat Scanner", "Image Converter"])

    with tab_scan:
        # Threat carousel
        st.markdown(
            "<p style='color:#ffffff; text-align:center; font-size:1.7rem; "
            "margin-bottom:0.2rem; font-weight:600;'>DETECTABLE THREAT CATEGORIES</p>",
            unsafe_allow_html=True,
        )
        render_carousel()

        st.markdown("<br>", unsafe_allow_html=True)

        # Upload section
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown(
                "<h3 style='color:#ffffff;'>Upload X-Ray Scan</h3>",
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Choose an X-ray image",
                type=["png", "jpg", "jpeg", "bmp"],
                label_visibility="collapsed",
                key="scan_uploader",
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded scan", use_container_width=True)

        with col_right:
            st.markdown(
                "<h3 style='color:#ffffff;'>Analysis Results</h3>",
                unsafe_allow_html=True,
            )

            if uploaded_file is not None:
                # Scanning animation (clearable placeholder)
                scan_placeholder = st.empty()
                with scan_placeholder.container():
                    st.markdown(
                        '<div class="scan-box">'
                        '<div class="scan-line"></div>'
                        '<div class="scan-animation">'
                        '<p>Analysing scan...</p>'
                        '</div></div>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(1.5)
                scan_placeholder.empty()

                # Load model and run prediction
                interpreter = load_model()
                predicted_class, confidence, probs = predict(interpreter, image)

                # Show results
                render_results(predicted_class, confidence, probs)
            else:
                st.markdown("""
                <div class="upload-area">
                    <p style="color:#ffffff; font-size:1.1rem; margin:0;">
                        Upload an X-ray image to begin analysis
                    </p>
                    <p style="color:#e2e8f0; font-size:0.85rem; margin-top:0.5rem;">
                        Supported formats: PNG, JPG, JPEG, BMP
                    </p>
                </div>
                """, unsafe_allow_html=True)

    with tab_convert:
        render_image_converter()

    # Footer
    st.markdown("""
    <div class="footer">
        Powered by TensorFlow Lite and EfficientNetB0 | DLOR Project
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
