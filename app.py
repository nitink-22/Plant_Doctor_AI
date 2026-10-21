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
# 1. PAGE CONFIGURATION & REFINED MINIMALIST CSS
# =============================================================================
st.set_page_config(
    page_title="Plant Doctor AI | Diagnostic Engine",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Top Navigation */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 32px;
        background: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 40px;
    }
    .nav-brand {
        font-weight: 700;
        font-size: 1.2rem;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    .nav-brand span {
        color: #10b981;
    }

    /* Hero Typography */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1.1;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin-bottom: 20px;
    }
    .hero-title span {
        color: #10b981;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 32px;
        max-width: 500px;
    }

    /* Feature Badges */
    .feature-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }
    .feature-badge {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #cbd5e1;
        font-weight: 500;
    }

    /* CLEAN TABS - Replacing the solid green block */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: #64748b !important;
        padding: 10px 4px !important;
        font-weight: 500;
        font-size: 0.95rem;
        border-radius: 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #10b981 !important;
        border-bottom: 2px solid #10b981 !important;
    }

    /* Diagnostic Result Cards */
    .result-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
    }
    .result-healthy {
        border-left: 4px solid #10b981;
    }
    .result-diseased {
        border-left: 4px solid #ef4444;
    }
    .result-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .result-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #f8fafc;
    }

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
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=3, dim=1)
        
    top_predictions = []
    for i in range(3):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item() * 100
        top_predictions.append((CLASS_NAMES[idx], prob))
        
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
    # Subtle Top Navigation
    st.markdown("""
        <div class="nav-bar">
            <div class="nav-brand">Plant Doctor <span>AI</span></div>
        </div>
    """, unsafe_allow_html=True)

    col_hero, col_scanner = st.columns([1.1, 0.9], gap="large")
    model = load_model()

    with col_hero:
        st.markdown("""
            <h1 class="hero-title">
                Intelligent Crop<br/>Pathology Engine
            </h1>
            <p class="hero-sub">
                Professional diagnostic platform for identifying plant pathogens. 
                Upload a sample to receive instant neural analysis backed by visual 
                attention maps and actionable treatment protocols.
            </p>
            <div class="feature-row">
                <div class="feature-badge">MobileNetV2 Architecture</div>
                <div class="feature-badge">Grad-CAM Explainability</div>
                <div class="feature-badge">38 Target Classes</div>
            </div>
        """, unsafe_allow_html=True)

    input_image = None

    with col_scanner:
        with st.container(border=True):
            st.markdown("<h4 style='margin-top:0; color:#f8fafc;'>Diagnostic Scanner</h4>", unsafe_allow_html=True)

            # Clean tabs with no emojis and modern underline styling
            tab_file, tab_cam = st.tabs(["Upload File", "Live Camera"])
            
            with tab_file:
                uploaded_file = st.file_uploader("Select image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
                if uploaded_file:
                    input_image = Image.open(uploaded_file)

            with tab_cam:
                cam_shot = st.camera_input("Capture live snapshot", label_visibility="collapsed")
                if cam_shot:
                    input_image = Image.open(cam_shot)

            if input_image:
                st.image(input_image, use_container_width=True)

                with st.spinner("Processing neural inference..."):
                    predictions, xai_image = predict_and_explain(model, input_image)

                top_class, top_conf = predictions[0]
                metadata = get_disease_metadata(top_class)
                is_healthy = "healthy" in top_class.lower()

                # Elegant Result Card
                status_class = "result-healthy" if is_healthy else "result-diseased"
                st.markdown(f"""
                    <div class="result-card {status_class}">
                        <div class="result-header">Primary Diagnosis</div>
                        <div class="result-value">{top_class}</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">
                            Confidence: {top_conf:.1f}% • Severity: {metadata['severity']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Detailed Analysis Section Below
    if input_image and 'predictions' in locals():
        st.write("")
        st.write("")
        st.subheader("Analysis & Protocol")
        
        col_xai, col_details = st.columns([1, 1], gap="large")

        with col_xai:
            with st.container(border=True):
                st.markdown("##### Visual Attention (Grad-CAM)")
                st.caption("Heatmap highlighting the specific leaf regions driving the model's prediction.")
                st.image(xai_image, use_container_width=True)

        with col_details:
            with st.container(border=True):
                st.markdown("##### Treatment Protocol")
                st.info(metadata["solution"])
                
                st.divider()

                st.markdown("##### Model Confidence Distribution")
                for cls_name, prob in predictions:
                    st.caption(f"{cls_name}")
                    st.progress(min(int(prob), 100))

if __name__ == "__main__":
    main()