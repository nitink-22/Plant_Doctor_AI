import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Doctor AI",
    page_icon="🌿",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. CLASS LABELS (38 Classes - Alphabetical Order Matching PyTorch ImageFolder)
# -----------------------------------------------------------------------------
CLASS_NAMES = [
    "Apple: Scab", 
    "Apple: Black Rot", 
    "Apple: Cedar Rust", 
    "Apple: Healthy",
    "Blueberry: Healthy", 
    "Cherry: Powdery Mildew", 
    "Cherry: Healthy",
    "Corn: Gray Leaf Spot", 
    "Corn: Common Rust", 
    "Corn: Northern Leaf Blight", 
    "Corn: Healthy",
    "Grape: Black Rot", 
    "Grape: Esca (Black Measles)", 
    "Grape: Leaf Blight", 
    "Grape: Healthy",
    "Orange: Citrus Greening (Haunglongbing)",
    "Peach: Bacterial Spot", 
    "Peach: Healthy",
    "Pepper (Bell): Bacterial Spot", 
    "Pepper (Bell): Healthy",
    "Potato: Early Blight", 
    "Potato: Late Blight", 
    "Potato: Healthy",
    "Raspberry: Healthy",
    "Soybean: Healthy",
    "Squash: Powdery Mildew",
    "Strawberry: Leaf Scorch", 
    "Strawberry: Healthy",
    "Tomato: Bacterial Spot", 
    "Tomato: Early Blight", 
    "Tomato: Late Blight", 
    "Tomato: Leaf Mold", 
    "Tomato: Septoria Leaf Spot", 
    "Tomato: Spider Mites", 
    "Tomato: Target Spot", 
    "Tomato: Yellow Leaf Curl Virus", 
    "Tomato: Mosaic Virus", 
    "Tomato: Healthy"
]

DISEASE_SOLUTIONS = {
    "Scab": "Apply copper-based fungicides or Captan/Mancozeb. Rake and destroy fallen infected leaves to prevent overwintering spores.",
    "Black Rot": "Prune out dead or diseased wood during dormancy. Apply copper-based fungicide before buds open.",
    "Cedar Rust": "Remove nearby galls on cedar trees if possible. Apply protective fungicides like Myclobutanil during bloom.",
    "Powdery Mildew": "Apply sulfur-based fungicides or neem oil. Ensure adequate sunlight and adequate air spacing between leaves.",
    "Gray Leaf Spot": "Use resistant hybrids, practice crop rotation with non-host crops, and apply triazole/strobilurin fungicides if severe.",
    "Common Rust": "Plant resistant crop varieties. Apply recommended foliar fungicides if infection starts early in the season.",
    "Northern Leaf Blight": "Rotate crops, plow under crop residue, and utilize resistant crop hybrids or approved fungicides.",
    "Esca (Black Measles)": "Remove infected branches down to healthy wood. Protect pruning wounds with wound sealants.",
    "Leaf Blight": "Apply protective copper fungicides early in the season and maintain good vine canopy management for airflow.",
    "Citrus Greening": "Control the psyllid insect vector using insecticides, remove infected trees immediately to stop spread, and apply balanced nutrition.",
    "Bacterial Spot": "Use disease-free seeds and copper-mancozeb sprays. Avoid overhead irrigation to minimize leaf wetness.",
    "Early Blight": "Apply copper or chlorothalonil fungicides. Mulch around plant base to prevent soil splash and rotate crops annually.",
    "Late Blight": "Apply protective fungicides (Chlorothalonil/Mancozeb) immediately. Destroy severely infected plants to prevent airborne spread.",
    "Leaf Mold": "Improve greenhouse/field ventilation to reduce humidity. Apply fungicides containing copper or chlorothalonil.",
    "Septoria Leaf Spot": "Remove infected lower leaves promptly. Apply fungicides and practice strict crop rotation.",
    "Spider Mites": "Spray leaves with water jets, insecticidal soap, or neem oil. Maintain good hydration to reduce mite pressure.",
    "Target Spot": "Maintain crop hygiene, lower moisture around leaves, and apply appropriate protective fungicides.",
    "Yellow Leaf Curl Virus": "Control whitefly populations using insecticidal soap, yellow sticky traps, and physical insect nets.",
    "Mosaic Virus": "No chemical cure. Remove and destroy infected plants immediately; sanitize hands and tools to prevent mechanical spread.",
    "Leaf Scorch": "Avoid overhead watering, remove diseased leaves in late summer, and ensure proper soil drainage.",
    "Healthy": "Plant is healthy! Maintain regular watering, optimal sunlight, and routine nutrient management."
}

# -----------------------------------------------------------------------------
# 3. MODEL LOADING (CACHED)
# -----------------------------------------------------------------------------
MODEL_PATH = "mobilenet_v2_baseline.pth"

@st.cache_resource
def load_model():
    # Recreate MobileNetV2 architecture with 38 output classes
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    
    # Load model state dict safely on CPU
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else:
        st.error(f"Model file '{MODEL_PATH}' not found in the root directory!")
    
    # CRITICAL: Set model to evaluation mode (disables dropout/batchnorm randomness)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
def transform_image(image: Image.Image) -> torch.Tensor:
    # Standard ImageNet normalization matching Colab training transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Convert image to RGB if grayscale or RGBA
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    return transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

# -----------------------------------------------------------------------------
# 5. INFERENCE FUNCTION
# -----------------------------------------------------------------------------
def predict(model: nn.Module, image_tensor: torch.Tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply Softmax to convert raw logits into valid probability percentages
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
    class_name = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    return class_name, confidence_score

# -----------------------------------------------------------------------------
# 6. STREAMLIT UI LAYOUT
# -----------------------------------------------------------------------------
def get_solution(class_name: str) -> str:
    for key, solution in DISEASE_SOLUTIONS.items():
        if key.lower() in class_name.lower():
            return solution
    return "Ensure optimal soil nutrition, proper irrigation, and consult local agricultural extensions if symptoms persist."

def main():
    st.title("🌿 AgriVision Doctor")
    st.markdown("Upload a clear leaf image to diagnose plant crop diseases and receive recommended treatment protocols.")

    # Load model
    model = load_model()

    # Sidebar info
    st.sidebar.header("⚙️ App Navigation & Info")
    st.sidebar.info(
        "**Architecture:** MobileNetV2\n\n"
        "**Dataset:** PlantVillage (38 Classes)\n\n"
        "**Deployment Engine:** PyTorch + Streamlit"
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload crop leaf image...", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

        with col2:
            st.subheader("🔍 Diagnostic Result")
            with st.spinner("Analyzing leaf pathology..."):
                input_tensor = transform_image(image)
                predicted_class, confidence = predict(model, input_tensor)

            # Display diagnosis tag
            if "healthy" in predicted_class.lower():
                st.success(f"**Diagnosis:** {predicted_class}")
            else:
                st.warning(f"**Diagnosis:** {predicted_class}")

            # Display confidence metric and progress bar
            st.metric(label="AI Model Confidence", value=f"{confidence:.2f}%")
            st.progress(min(int(confidence), 100))

            st.divider()

            # Display recommended treatment
            st.subheader("💡 Recommended Treatment / Action")
            solution = get_solution(predicted_class)
            st.info(solution)

if __name__ == "__main__":
    main()