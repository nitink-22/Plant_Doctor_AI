import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import streamlit as st

# =============================================================================
# 1. PAGE CONFIGURATION & MODERN SAAS CSS STYLING
# =============================================================================
st.set_page_config(
    page_title="Crop Disease Detection | Professional AI Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS directly inspired by modern SaaS landing page UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b1320 0%, #111c2e 50%, #0a111d 100%);
        color: #f8fafc;
    }

    /* Top Navigation Header Bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 28px;
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 100px;
        margin-bottom: 32px;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 800;
        font-size: 1.25rem;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .nav-brand-icon {
        background: #10b981;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    .nav-links {
        display: flex;
        gap: 28px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .nav-btn {
        background: #10b981;
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.85rem;
        text-decoration: none;
    }

    /* Hero Left Column Copy */
    .social-proof {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 6px 16px;
        border-radius: 100px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.12;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin-bottom: 18px;
    }
    .hero-title span {
        color: #10b981;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 28px;
        max-width: 520px;
    }

    /* Feature Pills */
    .pill-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .pill-chip {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 8px 16px;
        border-radius: 100px;
        font-size: 0.82rem;
        color: #e2e8f0;
        font-weight: 500;
    }

    /* Floating Scanner Card Widget */
    .scanner-card-header {
        text-align: center;
        padding-bottom: 12px;
    }
    .scanner-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .scanner-card-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Results Alert Box Inside Card */
    .alert-box-diseased {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 14px 18px;
        color: #991b1b;
        margin-top: 16px;
    }
    .alert-box-healthy {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 14px 18px;
        color: #166534;
        margin-top: 16px;
    }
    .alert-title {
        font-weight: 700;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .alert-meta {
        font-size: 0.82rem;
        opacity: 0.85;
        margin-top: 2px;
    }

    /* Streamlit Tab Override for White Scanner Card */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 8px 16px;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: #ffffff !important;
        border-color: #10b981 !important;
    }

    /* Hide Default Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CLASS LABELS & KNOWLEDGE BASE
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
    "Scab": {"solution": "Apply protective copper-based fungicides or Captan/Mancozeb. Remove and destroy fallen infected leaf debris.", "severity": "Moderate"},
    "Black Rot": {"solution": "Prune away dead or cankered wood during dormancy. Apply copper fungicides before spring bud bloom.", "severity": "High"},
    "Cedar Rust": {"solution": "Remove nearby galls on cedar trees. Apply protective fungicides like Myclobutanil during early growth phases.", "severity": "Moderate"},
    "Powdery Mildew": {"solution": "Apply sulfur-based sprays or organic neem oil. Improve sunlight exposure and canopy air ventilation.", "severity": "Low-Moderate"},
    "Gray Leaf Spot": {"solution": "Practice strategic crop rotation with non-host crops and apply targeted triazole fungicides if severe.", "severity": "High"},
    "Common Rust": {"solution": "Deploy rust-resistant hybrids and apply recommended foliar fungicides if infection initiates early.", "severity": "Moderate"},
    "Northern Leaf Blight": {"solution": "Rotate field crops, plow under post-harvest residue, and apply approved preventive fungicides.", "severity": "High"},
    "Esca (Black Measles)": {"solution": "Prune infected branches well down into healthy wood. Coat pruning cuts with protective wound sealants.", "severity": "High"},
    "Leaf Blight": {"solution": "Apply copper fungicides early in the season and manage canopy density to reduce ambient humidity.", "severity": "Moderate"},
    "Citrus Greening": {"solution": "Control psyllid insect vectors using systemic insecticides. Remove severely infected trees immediately.", "severity": "Critical"},
    "Bacterial Spot": {"solution": "Use certified disease-free seed stock and copper-mancozeb sprays. Avoid overhead spray irrigation.", "severity": "High"},
    "Early Blight": {"solution": "Spray copper or chlorothalonil fungicides. Mulch around the plant base to prevent soil splash.", "severity": "Moderate"},
    "Late Blight": {"solution": "Apply protective fungicides (Chlorothalonil/Mancozeb) immediately. Destroy severely blighted plants.", "severity": "Critical"},
    "Leaf Mold": {"solution": "Increase greenhouse ventilation and lower relative humidity. Apply copper-based fungicides at first sight.", "severity": "Moderate"},
    "Septoria Leaf Spot": {"solution": "Prune out infected lower leaves. Apply fungicides regularly and enforce strict annual crop rotation.", "severity": "Moderate"},
    "Spider Mites": {"solution": "Apply high-pressure water jets, insecticidal soap, or horticultural neem oil. Maintain adequate hydration.", "severity": "Low"},
    "Target Spot": {"solution": "Maintain field hygiene, lower moisture around foliage, and apply broad-spectrum protective fungicides.", "severity": "Moderate"},
    "Yellow Leaf Curl Virus": {"solution": "Manage whitefly vectors with insecticidal soap, yellow sticky cards, and fine physical mesh netting.", "severity": "High"},
    "Mosaic Virus": {"solution": "No chemical cure available. Immediately rogue and burn infected plants. Disinfect all tools and hands.", "severity": "Critical"},
    "Leaf Scorch": {"solution": "Avoid overhead leaf watering, trim diseased foliage in late summer, and ensure well-draining soil.", "severity": "Low-Moderate"},
    "Healthy": {"solution": "No pathological symptoms detected. Continue routine irrigation, proper fertilization, and monitoring.", "severity": "None"}
}

# =============================================================================
# 3. GRAD-CAM EXPLAINABLE AI ENGINE
# =============================================================================
class GradCAM:
    """Computes Class Activation Maps for MobileNetV2 without external heavy dependencies."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class]
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, target_class

def generate_overlay(original_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Generates Jet colormap heatmap overlay scaled to target image."""
    img_resized = original_pil.resize((224, 224)).convert("RGB")
    cam_resized = Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)
    
    cam_np = np.array(cam_resized) / 255.0
    colormap = cm.get_cmap("jet")
    heatmap_rgb = colormap(cam_np)[:, :, :3]
    
    orig_np = np.array(img_resized) / 255.0
    overlay = (1 - alpha) * orig_np + alpha * heatmap_rgb
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(overlay)

# =============================================================================
# 4. MODEL LOADING & INFERENCE
# =============================================================================
MODEL_PATH = "mobilenet_v2_baseline.pth"

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
    else:
        st.warning(f"Model file '{MODEL_PATH}' was not found in root directory.")
        
    model.eval()
    return model

def transform_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def predict_and_explain(model, image: Image.Image):
    tensor = transform_image(image)
    
    # Forward Pass
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=3, dim=1)
        
    top_predictions = []
    for i in range(3):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item() * 100
        top_predictions.append((CLASS_NAMES[idx], prob))
        
    # Grad-CAM Hook Execution
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    cam_heatmap, _ = grad_cam.generate_heatmap(tensor)
    xai_image = generate_overlay(image, cam_heatmap)
    
    return top_predictions, xai_image

def get_disease_metadata(class_name: str):
    for key, data in DISEASE_KNOWLEDGE_BASE.items():
        if key.lower() in class_name.lower():
            return data
    return {"solution": "Consult local agricultural extension specialists.", "severity": "Unknown"}

# =============================================================================
# 5. MAIN DASHBOARD INTERFACE
# =============================================================================
def main():
    # Top Navigation Bar
    st.markdown("""
        <div class="nav-bar">
            <div class="nav-brand">
                <div class="nav-brand-icon">🌿</div>
                Crop Disease Detection
            </div>
            <div class="nav-links">
                <span>Disease ID</span>
                <span>Crop Pathology</span>
                <span>Disease Diagnosis</span>
                <span>Pest Scanner</span>
            </div>
            <a href="#scan-section" class="nav-btn">Detect Disease Now</a>
        </div>
    """, unsafe_allow_html=True)

    # Hero Section Split (Left Copy / Right Card)
    col_hero_left, col_hero_right = st.columns([1.1, 0.9], gap="large")

    model = load_model()

    with col_hero_left:
        st.markdown("""
            <div class="social-proof">
                <span>👨‍🌾</span> Trusted by 10,000+ agricultural practitioners worldwide
            </div>
            <h1 class="hero-title">
                Professional Crop<br/>Disease Detection<br/>with <span>98% Accuracy</span>
            </h1>
            <p class="hero-sub">
                Specialized AI platform for identifying plant diseases and pathogens. 
                Get instant diagnosis, Explainable AI (Grad-CAM) visual attention maps, 
                and actionable treatment protocols.
            </p>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="pill-container">
                <div class="pill-chip">✓ 98% Model Accuracy</div>
                <div class="pill-chip">⚡ Instant Neural Results</div>
                <div class="pill-chip">🔥 Grad-CAM XAI Supported</div>
                <div class="pill-chip">🛡️ 38 Crop Pathologies</div>
            </div>
        """, unsafe_allow_html=True)

    # Right Column: The Interactive Scanner Card Widget
    input_image = None

    with col_hero_right:
        with st.container(border=True):
            st.markdown("""
                <div class="scanner-card-header">
                    <h3 class="scanner-card-title">Scan Your Crops</h3>
                    <p class="scanner-card-sub">Point your camera at leaves or upload an image file</p>
                </div>
            """, unsafe_allow_html=True)

            tab_file, tab_cam = st.tabs(["📁 Upload Image File", "📸 Live Camera"])
            
            with tab_file:
                uploaded_file = st.file_uploader("Select crop leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
                if uploaded_file:
                    input_image = Image.open(uploaded_file)

            with tab_cam:
                cam_shot = st.camera_input("Capture leaf snapshot", label_visibility="collapsed")
                if cam_shot:
                    input_image = Image.open(cam_shot)

            if input_image:
                st.image(input_image, use_container_width=True)

                with st.spinner("Analyzing crop pathology & generating Grad-CAM heatmap..."):
                    predictions, xai_image = predict_and_explain(model, input_image)

                top_class, top_conf = predictions[0]
                metadata = get_disease_metadata(top_class)
                is_healthy = "healthy" in top_class.lower()

                # Inline Diagnosis Alert Box Matching Reference Screenshot
                if is_healthy:
                    st.markdown(f"""
                        <div class="alert-box-healthy">
                            <div class="alert-title">✅ Diagnosis: {top_class}</div>
                            <div class="alert-meta">Model Confidence: {top_conf:.1f}% • Plant Status: Healthy</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="alert-box-diseased">
                            <div class="alert-title">⚠️ Warning: {top_class}</div>
                            <div class="alert-meta">Model Confidence: {top_conf:.1f}% • Severity Level: {metadata['severity']}</div>
                        </div>
                    """, unsafe_allow_html=True)

    # Detailed XAI & Treatment Breakdown Below
    if input_image and 'predictions' in locals():
        st.divider()
        st.markdown("<h3 style='margin-bottom: 20px;'>🔬 Detailed Diagnostic Breakdown & XAI Attention</h3>", unsafe_allow_html=True)

        col_xai, col_details = st.columns([1, 1], gap="large")

        with col_xai:
            with st.container(border=True):
                st.markdown("#### Explainable AI (Grad-CAM Heatmap)")
                st.caption("Red and yellow regions indicate key diagnostic lesion patterns identified by the model.")
                st.image(xai_image, use_container_width=True)

        with col_details:
            with st.container(border=True):
                st.markdown("#### Clinical Treatment & Action Plan")
                st.info(metadata["solution"])

                st.divider()

                st.markdown("#### Top Class Probability Distribution")
                for cls_name, prob in predictions:
                    st.text(f"{cls_name} — {prob:.1f}%")
                    st.progress(min(int(prob), 100))

if __name__ == "__main__":
    main()