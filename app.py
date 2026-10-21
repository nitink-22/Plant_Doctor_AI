import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from rembg import remove, new_session
import numpy as np
import cv2

# -------------------------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS 
# -------------------------------------------------------------------------
# Removed the emoji page_icon
st.set_page_config(page_title="Plant Doctor AI", layout="wide")

st.markdown("""
    <style>
    /* Make the deep background a solid dark color */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Box the main content into a sleek, centralized dashboard card */
    .block-container {
        max-width: 1300px !important;
        background-color: #1A1C23;
        padding: 3rem 4rem !important;
        border-radius: 20px;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.4);
        margin-top: 3vh;
        margin-bottom: 3vh;
    }

    /* Smooth rounded corners for all images */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Style and center the main title */
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 800;
        color: #4CAF50;
        margin-bottom: 0px;
        text-align: center;
        letter-spacing: 1px;
    }
    
    /* Style and center the subheader */
    .sub-title {
        font-size: 1.1rem;
        color: #A0A0A0;
        margin-bottom: 40px;
        text-align: center;
    }
    
    /* Clean up the sidebar */
    .css-1d391kg {
        background-color: #121418;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2. EXACT ALPHABETICAL 38 CLASS LABELS & SOLUTIONS
# -------------------------------------------------------------------------
CLASS_NAMES = [
    "Apple: Scab", "Apple: Black Rot", "Apple: Cedar Rust", "Apple: Healthy",
    "Blueberry: Healthy",
    "Cherry: Powdery Mildew", "Cherry: Healthy",
    "Corn: Gray Leaf Spot", "Corn: Common Rust", "Corn: Northern Leaf Blight", "Corn: Healthy",
    "Grape: Black Rot", "Grape: Black Measles", "Grape: Leaf Blight", "Grape: Healthy",
    "Orange: Citrus Greening",
    "Peach: Bacterial Spot", "Peach: Healthy",
    "Pepper (Bell): Bacterial Spot", "Pepper (Bell): Healthy",
    "Potato: Early Blight", "Potato: Late Blight", "Potato: Healthy",
    "Raspberry: Healthy",
    "Soybean: Healthy",
    "Squash: Powdery Mildew",
    "Strawberry: Leaf Scorch", "Strawberry: Healthy",
    "Tomato: Bacterial Spot", "Tomato: Early Blight", "Tomato: Late Blight",
    "Tomato: Leaf Mold", "Tomato: Septoria Leaf Spot", "Tomato: Spider Mites",
    "Tomato: Target Spot", "Tomato: Yellow Leaf Curl Virus", "Tomato: Mosaic Virus", "Tomato: Healthy"
]

DISEASE_SOLUTIONS = {
    "Scab": "Apply fungicides like Captan or Mancozeb. Rake and destroy fallen leaves.",
    "Black Rot": "Prune out dead or diseased wood. Apply a copper-based fungicide before buds open.",
    "Cedar Rust": "Remove nearby cedar hosts if possible. Apply preventative fungicides containing myclobutanil.",
    "Powdery Mildew": "Increase air circulation through pruning. Apply sulfur or potassium bicarbonate sprays.",
    "Gray Leaf Spot": "Use crop rotation. Apply foliar fungicides like strobilurins if symptoms are severe.",
    "Common Rust": "Plant rust-resistant varieties next season. Apply fungicide early if rust pustules cover leaves.",
    "Northern Leaf Blight": "Rotate crops away from corn for 1-2 years. Fungicide is rarely economical unless very wet.",
    "Black Measles": "Also known as Esca. Prune out infected wood. No chemical cure exists; focus on vine health.",
    "Leaf Blight": "Ensure proper drainage and avoid overhead watering. Copper fungicides can slow the spread.",
    "Citrus Greening": "Incurable bacterial disease spread by psyllids. Remove and destroy infected trees immediately.",
    "Bacterial Spot": "Apply copper-based bactericides early in the season. Avoid overhead irrigation.",
    "Early Blight": "Apply chlorothalonil or copper fungicides. Practice a 3-year crop rotation.",
    "Late Blight": "Aggressive and highly contagious. Apply fungicides immediately. Destroy infected plants.",
    "Leaf Scorch": "Avoid overhead watering. Remove infected foliage and ensure proper plant spacing.",
    "Leaf Mold": "Reduce greenhouse humidity. Improve ventilation and apply appropriate fungicides if severe.",
    "Septoria Leaf Spot": "Mulch soil to prevent splashing. Water at the base. Apply fungicidal sprays if spotted early.",
    "Spider Mites": "Spray leaves with water to dislodge mites. Introduce predatory mites or use horticultural oils (Neem).",
    "Target Spot": "Apply broad-spectrum fungicides. Ensure crops have adequate airflow.",
    "Yellow Leaf Curl Virus": "Spread by whiteflies. Use reflective mulches and insecticidal soaps. Destroy infected plants.",
    "Mosaic Virus": "No cure. Highly contagious via touch or tools. Uproot and burn infected plants immediately. Wash hands and tools."
}

# -------------------------------------------------------------------------
# 3. GRAD-CAM ENGINE (AI X-Ray)
# -------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

# -------------------------------------------------------------------------
# 4. MODEL CONFIGURATION & LOADING
# -------------------------------------------------------------------------
@st.cache_resource
def load_agtech_model():
    model = models.resnet50(weights=None) 
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 38)
    
    try:
        model.load_state_dict(torch.load("enterprise_plant_model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("Error: 'enterprise_plant_model.pth' not found. Please place it in the same folder as app.py.")
        st.stop()
        
    model.eval()
    return model

model = load_agtech_model()
cam_engine = GradCAM(model, model.layer4[-1])

@st.cache_resource
def load_rembg_session():
    return new_session("u2net") # Full model with superior leaf edge detection

my_session = load_rembg_session()

input_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------------------------------------------------
# 5. STREAMLIT INTERFACE UI & SIDEBAR DASHBOARD
# -------------------------------------------------------------------------
st.markdown('<p class="main-title">Plant Doctor AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Production-ready AI system with automated background removal and Grad-CAM visual diagnostics.</p>', unsafe_allow_html=True)
st.markdown("---")

st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #4CAF50; margin-bottom: 0px;">Control Panel</h2>
        <p style="color: #888; font-size: 0.9rem;">Configure diagnostic parameters</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Image Source")
input_mode = st.sidebar.radio("Select how to provide the leaf image:", ["Upload Local Image", "Use Live Device Camera"], label_visibility="collapsed")

st.sidebar.markdown("### Processing Options")
enable_bg_removal = st.sidebar.checkbox("Auto-Isolate Leaf", value=True, help="Automatically removes the background to help the AI focus solely on the leaf structure.")

st.sidebar.markdown("---")

image_file = None
if input_mode == "Upload Local Image":
    image_file = st.sidebar.file_uploader("Drop a leaf photo here...", type=["jpg", "jpeg", "png"])
else:
    image_file = st.sidebar.camera_input("Position leaf clearly in front of device camera")

st.sidebar.markdown("---")

st.sidebar.success("""
**[ONLINE] System Status** **Engine:** ResNet-50 Vision  
**Processing:** Local Inference  
**Classes:** 38 Pathogen Profiles
""")

with st.sidebar.expander("View Supported Crops (14)"):
    st.markdown("""
    * Apple
    * Blueberry
    * Cherry
    * Corn
    * Grape
    * Orange
    * Peach
    * Pepper (Bell)
    * Potato
    * Raspberry
    * Soybean
    * Squash
    * Strawberry
    * Tomato
    """)

st.sidebar.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.8rem;">
        Plant Doctor AI v1.0.0<br>
        <i>Powered by Deep Learning</i>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 6. PROCESSING & PREDICTION PIPELINE
# -------------------------------------------------------------------------
if image_file is not None:
    raw_image = Image.open(image_file).convert("RGB")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Original Capture**")
        st.image(raw_image, use_container_width=True)
        
    if enable_bg_removal:
        with st.spinner("Isolating leaf..."):
            my_session = new_session("u2netp")
            # NEW CODE (Preserves leaf structure):
            bg_removed = remove(raw_image, session=my_session)
            
            processed_image = Image.new("RGB", bg_removed.size, (255, 255, 255))
            processed_image.paste(bg_removed, mask=bg_removed.split()[3])
            
        with col2:
            st.markdown("**2. Isolated Matrix**")
            st.image(processed_image, width="stretch")
    else:
        processed_image = raw_image
        with col2:
            st.info("Isolation skipped.")

    with st.spinner("Analyzing botanical structures..."):
        tensor_img = input_transforms(processed_image).unsqueeze(0)
        
        outputs = model(tensor_img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, class_idx = torch.max(probabilities, 0)
        
        heatmap = cam_engine.generate_heatmap(tensor_img, class_idx.item())
        
        orig_width, orig_height = processed_image.size
        heatmap_resized = cv2.resize(heatmap, (orig_width, orig_height))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        processed_img_np = np.array(processed_image)
        overlay = cv2.addWeighted(processed_img_np, 0.5, heatmap_color, 0.5, 0)
        
        with col3:
            st.markdown("**3. AI Focus (Grad-CAM)**")
            st.image(overlay, use_container_width=True)
                        
    # -------------------------------------------------------------------------
    # 7. DIAGNOSTIC RESULTS DISPLAY 
    # -------------------------------------------------------------------------
    st.markdown("---")
    
    detected_condition = CLASS_NAMES[class_idx.item()]
    confidence_score = confidence.item() * 100
    
    metric_col1, metric_col2 = st.columns([2, 1])
    with metric_col1:
        st.subheader("Diagnostic Report")
    with metric_col2:
        st.metric(label="AI Confidence Level", value=f"{confidence_score:.2f}%")
    
    if "Healthy" in detected_condition:
        st.success(f"**Status:** {detected_condition}")
        st.write("**Recommendation:** Continue current watering and nutrient schedules. No pathogens detected.")
    else:
        st.error(f"**Pathogen Detected:** {detected_condition}")
        
        solution_found = False
        for key, solution in DISEASE_SOLUTIONS.items():
            if key in detected_condition:
                st.warning(f"**Recommended Treatment:** {solution}")
                solution_found = True
                break
                
        if not solution_found:
            st.warning("**Recommended Treatment:** Isolate the plant. Consult local agricultural extension for specific regional treatments.")