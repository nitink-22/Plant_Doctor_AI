import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# =============================================================================
# 1. PAGE CONFIGURATION & GLOBAL SETUP
# =============================================================================
st.set_page_config(
    page_title="Plant Doctor AI - Intelligent Crop Pathology",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. CUSTOM CSS (Modern Glassmorphism & Micro-Interactions)
# =============================================================================
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Background */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Glassmorphic Card Containers */
    .custom-card {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    }

    /* Status Badges */
    .status-badge-healthy {
        display: inline-flex;
        align-items: center;
        background: rgba(35, 134, 54, 0.2);
        border: 1px solid #238636;
        color: #3fb950;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .status-badge-diseased {
        display: inline-flex;
        align-items: center;
        background: rgba(218, 54, 51, 0.2);
        border: 1px solid #da3633;
        color: #f85149;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Metric Box Highlights */
    .metric-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f6fc;
    }

    /* Header Banner Styling */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0px 24px 0px;
        border-bottom: 1px solid #30363d;
        margin-bottom: 28px;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2ea043, #58a6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* System Status Indicator */
    .system-status {
        display: inline-flex;
        align-items: center;
        background: #161b22;
        border: 1px solid #30363d;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #3fb950;
    }
    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #238636;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 8px #238636;
    }

    /* Top-3 Confidence Bars */
    .top-pred-item {
        margin-bottom: 12px;
    }
    .top-pred-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }

    /* Streamlit Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 8px;
        color: #8b949e;
        border: 1px solid #30363d;
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important;
        color: #ffffff !important;
        border-color: #2ea043 !important;
    }

    /* Hide Streamlit default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. CLASS LABELS & PATHOLOGY KNOWLEDGE BASE
# =============================================================================
CLASS_NAMES = [
    "Apple: Scab", "Apple: Black Rot", "Apple: Cedar Rust", "Apple: Healthy",
    "Blueberry: Healthy", "Cherry: Powdery Mildew", "Cherry: Healthy",
    "Corn: Gray Leaf Spot", "Corn: Common Rust", "Corn: Northern Leaf Blight", "Corn: Healthy",
    "Grape: Black Rot", "Grape: Esca (Black Measles)", "Grape: Leaf Blight", "Grape: Healthy",
    "Orange: Citrus Greening", "Peach: Bacterial Spot", "Peach: Healthy",
    "Pepper (Bell): Bacterial Spot", "Pepper (Bell): Healthy",
    "Potato: Early Blight", "Potato: Late Blight", "Potato: Healthy",
    "Raspberry: Healthy", "Soybean: Healthy", "Squash: Powdery Mildew",
    "Strawberry: Leaf Scorch", "Strawberry: Healthy",
    "Tomato: Bacterial Spot", "Tomato: Early Blight", "Tomato: Late Blight", 
    "Tomato: Leaf Mold", "Tomato: Septoria Leaf Spot", "Tomato: Spider Mites", 
    "Tomato: Target Spot", "Tomato: Yellow Leaf Curl Virus", "Tomato: Mosaic Virus", 
    "Tomato: Healthy"
]

DISEASE_KNOWLEDGE_BASE = {
    "Scab": {
        "solution": "Apply protective copper-based fungicides or Captan/Mancozeb. Remove and burn fallen infected leaves.",
        "severity": "Moderate",
        "action": "Fungicide Spray & Sanitation"
    },
    "Black Rot": {
        "solution": "Prune out dead or diseased wood during winter dormancy. Apply copper fungicides prior to leaf bloom.",
        "severity": "High",
        "action": "Immediate Pruning Required"
    },
    "Cedar Rust": {
        "solution": "Remove nearby cedar tree galls if possible. Apply protective Myclobutanil fungicides early in spring.",
        "severity": "Moderate",
        "action": "Protective Spraying"
    },
    "Powdery Mildew": {
        "solution": "Apply organic neem oil or sulfur sprays. Increase sunlight and pruning to improve canopy airflow.",
        "severity": "Low to Moderate",
        "action": "Airflow & Organic Spray"
    },
    "Gray Leaf Spot": {
        "solution": "Practice annual crop rotation with non-host plants. Apply triazole or strobilurin fungicides if severe.",
        "severity": "High",
        "action": "Crop Rotation & Fungicide"
    },
    "Common Rust": {
        "solution": "Deploy rust-resistant hybrids. Apply foliar fungicides early if infection starts early in the growing season.",
        "severity": "Moderate",
        "action": "Foliar Fungicide Application"
    },
    "Northern Leaf Blight": {
        "solution": "Rotate field crops, deeply plow post-harvest debris, and use approved resistant hybrids.",
        "severity": "High",
        "action": "Debris Removal & Resistance"
    },
    "Esca (Black Measles)": {
        "solution": "Prune infected branches down to healthy tissue. Coat large pruning wounds with protective paint/sealant.",
        "severity": "High",
        "action": "Surgical Pruning"
    },
    "Leaf Blight": {
        "solution": "Apply copper-based protective fungicides early and manage crop density to reduce ambient humidity.",
        "severity": "Moderate",
        "action": "Humidity Reduction & Spray"
    },
    "Citrus Greening": {
        "solution": "Control psyllid insect vectors using systemic insecticides. Immediately rogue infected trees to prevent orchard loss.",
        "severity": "Critical",
        "action": "Vector Control & Eradication"
    },
    "Bacterial Spot": {
        "solution": "Use certified disease-free seeds and copper-mancozeb tank mixes. Strictly avoid overhead irrigation.",
        "severity": "High",
        "action": "Drip Irrigation & Bactericide"
    },
    "Early Blight": "Spray copper or chlorothalonil fungicides. Mulch around plant base to prevent soil splash.",
    "Early Blight": {
        "solution": "Apply chlorothalonil or copper fungicides. Mulch ground around plant base to prevent splash spread.",
        "severity": "Moderate",
        "action": "Base Mulching & Fungicide"
    },
    "Late Blight": {
        "solution": "Apply protective systemic fungicides (Mancozeb/Chlorothalonil) immediately. Destroy infected plants to prevent airborne spread.",
        "severity": "Critical",
        "action": "Emergency Quarantine / Spray"
    },
    "Leaf Mold": {
        "solution": "Increase greenhouse ventilation and keep humidity below 85%. Apply copper fungicides at first notice.",
        "severity": "Moderate",
        "action": "Ventilation Control"
    },
    "Septoria Leaf Spot": {
        "solution": "Remove infected lower leaves immediately. Apply fungicides and practice strict 3-year crop rotation.",
        "severity": "Moderate",
        "action": "Lower Leaf Pruning"
    },
    "Spider Mites": {
        "solution": "Apply high-pressure water jets, insecticidal soap, or horticultural neem oil. Maintain proper plant hydration.",
        "severity": "Low",
        "action": "Mite Soap / Hydration"
    },
    "Target Spot": {
        "solution": "Maintain optimal plant spacing, control humidity, and apply broad-spectrum protective fungicides.",
        "severity": "Moderate",
        "action": "Spacing & Protection"
    },
    "Yellow Leaf Curl Virus": {
        "solution": "Control whitefly vectors using insecticidal soaps, yellow sticky traps, and physical protective mesh screens.",
        "severity": "High",
        "action": "Insect Vector Elimination"
    },
    "Mosaic Virus": {
        "solution": "No chemical cure available. Rogue and incinerate infected plants immediately. Disinfect all tools and hands.",
        "severity": "Critical",
        "action": "Immediate Plant Removal"
    },
    "Leaf Scorch": {
        "solution": "Avoid overhead leaf watering, trim dead foliage in late summer, and ensure well-draining soil conditions.",
        "severity": "Low to Moderate",
        "action": "Drip Irrigation Adjustment"
    },
    "Healthy": {
        "solution": "No pathological symptoms identified. Continue balanced irrigation, routine fertilization, and monitoring.",
        "severity": "None (Healthy)",
        "action": "Standard Maintenance"
    }
}

# =============================================================================
# 4. MODEL INFERENCE ENGINE
# =============================================================================
MODEL_PATH = "mobilenet_v2_baseline.pth"

@st.cache_resource
def load_trained_model():
    """Loads and caches the PyTorch MobileNetV2 architecture with evaluation weights."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        except Exception as e:
            st.error(f"Error loading model state dict: {e}")
    else:
        st.warning(f"⚠️ Weights file '{MODEL_PATH}' not found in root directory. Running in architecture-only mode.")
    
    model.eval()  # Disables Dropout & locks BatchNorm statistics
    return model

def build_transform_pipeline():
    """Standard PyTorch evaluation pipeline matching ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def run_inference(model: nn.Module, image: Image.Image):
    """Performs model forward pass and computes top-3 prediction probabilities."""
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    transform = build_transform_pipeline()
    tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    
    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)
        
    top_predictions = []
    for i in range(3):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item() * 100
        top_predictions.append((CLASS_NAMES[idx], prob))
        
    return top_predictions

def lookup_knowledge(class_name: str):
    """Retrieves pathology details from knowledge base matching disease substring."""
    for key, data in DISEASE_KNOWLEDGE_BASE.items():
        if key.lower() in class_name.lower():
            return data
    return {
        "solution": "Consult local agricultural extension officers for localized treatment options.",
        "severity": "Unknown",
        "action": "Expert Consultation Required"
    }

# =============================================================================
# 5. SIDEBAR LAYOUT
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='margin-bottom:0;'>🌿 Plant Doctor AI</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#8b949e; font-size:0.85rem;'>AI-Powered Edge Pathology Suite</p>", unsafe_allow_html=True)
        st.divider()

        st.markdown("#### ⚙️ Architecture Metrics")
        st.markdown("""
            <div style="background:#161b22; padding:12px; border-radius:8px; border:1px solid #30363d; margin-bottom:10px;">
                <span style="color:#8b949e; font-size:0.8rem;">BACKBONE</span><br/>
                <strong style="color:#f0f6fc;">MobileNetV2 (3.5M Params)</strong>
            </div>
            <div style="background:#161b22; padding:12px; border-radius:8px; border:1px solid #30363d; margin-bottom:10px;">
                <span style="color:#8b949e; font-size:0.8rem;">TARGET CLASSES</span><br/>
                <strong style="color:#f0f6fc;">38 Crop Pathology Classes</strong>
            </div>
            <div style="background:#161b22; padding:12px; border-radius:8px; border:1px solid #30363d;">
                <span style="color:#8b949e; font-size:0.8rem;">INFERENCE DEVICE</span><br/>
                <strong style="color:#f0f6fc;">PyTorch CPU Engine</strong>
            </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📋 Operational Checklist")
        st.caption("✔️ Center single plant leaf in frame.")
        st.caption("✔️ Ensure bright, non-glare lighting.")
        st.caption("✔️ Keep background uniform if possible.")

# =============================================================================
# 6. MAIN APP INTERFACE
# =============================================================================
def main():
    render_sidebar()

    # Top Header Bar
    st.markdown("""
        <div class="app-header">
            <div>
                <h1 class="header-title">Plant Doctor AI</h1>
                <p class="header-subtitle">Real-time agricultural computer vision for instant leaf disease diagnosis</p>
            </div>
            <div class="system-status">
                <span class="status-dot"></span> System Ready
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Load Model Resource
    model = load_trained_model()

    # File Input Mode Selection
    input_tab1, input_tab2 = st.tabs(["📁 Upload Image File", "📸 Take Live Camera Snapshot"])
    
    selected_image = None

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Select crop leaf image file", 
            type=["jpg", "jpeg", "png"], 
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)

    with input_tab2:
        camera_snapshot = st.camera_input(
            "Capture leaf photo directly", 
            label_visibility="collapsed"
        )
        if camera_snapshot is not None:
            selected_image = Image.open(camera_snapshot)

    st.write("")

    # Main Analysis Dashboard
    if selected_image is not None:
        # Layout Split: 45% Image Frame, 55% Analysis Dashboard
        col_img, col_results = st.columns([0.85, 1.15], gap="large")

        with col_img:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0;'>📷 Target Image</h4>", unsafe_allow_html=True)
            
            # Display image with clean rounded framing
            st.image(selected_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_results:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0;'>📊 Diagnostic Breakdown</h4>", unsafe_allow_html=True)

            with st.spinner("Executing neural model evaluation..."):
                predictions = run_inference(model, selected_image)

            top_class, top_conf = predictions[0]
            is_healthy = "healthy" in top_class.lower()
            knowledge = lookup_knowledge(top_class)

            # Primary Status Badge
            if is_healthy:
                st.markdown(f'<div class="status-badge-healthy">✅ {top_class}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-badge-diseased">⚠️ {top_class}</div>', unsafe_allow_html=True)

            st.write("")
            st.write("")

            # Metric Bar Highlights
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Model Confidence</div>
                        <div class="metric-value">{top_conf:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Risk Severity</div>
                        <div class="metric-value" style="font-size:1.2rem; margin-top:5px;">{knowledge['severity']}</div>
                    </div>
                """, unsafe_allow_html=True)
            with m3:
                reliability = "High" if top_conf > 75 else "Moderate" if top_conf > 45 else "Low"
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Reliability</div>
                        <div class="metric-value" style="font-size:1.2rem; margin-top:5px;">{reliability}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Top-3 Model Probability Breakdown
            st.markdown("##### 📈 Top Probability Distribution")
            for cls_name, prob in predictions:
                st.markdown(f"""
                    <div class="top-pred-item">
                        <div class="top-pred-header">
                            <span style="color:#e6edf3;">{cls_name}</span>
                            <span style="color:#8b949e; font-weight:600;">{prob:.1f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(min(int(prob), 100))

            st.divider()

            # Actionable Solution Card
            st.markdown("##### 💡 Recommended Clinical Action")
            st.markdown(f"""
                <div style="background: rgba(40, 50, 70, 0.4); border-left: 4px solid #58a6ff; padding: 16px; border-radius: 8px;">
                    <p style="margin:0 0 6px 0; color:#58a6ff; font-weight:600; font-size:0.9rem;">
                        PRIMARY ACTION: {knowledge['action']}
                    </p>
                    <p style="margin:0; color:#c9d1d9; font-size:0.92rem; line-height:1.5;">
                        {knowledge['solution']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Placeholder view when no image is uploaded yet
        st.markdown("""
            <div class="custom-card" style="text-align: center; padding: 48px 24px;">
                <p style="font-size: 3rem; margin-bottom: 8px;">🍃</p>
                <h3 style="margin-top: 0; color: #f0f6fc;">Awaiting Input Image</h3>
                <p style="color: #8b949e; max-width: 500px; margin: 0 auto;">
                    Please upload a plant leaf image or use the live camera tab above to initiate real-time AI diagnosis.
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()