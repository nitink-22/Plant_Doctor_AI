# Plant Doctor AI

Production-ready AI system with automated background removal and Grad-CAM visual diagnostics.

Plant Doctor AI is an enterprise-grade Streamlit application designed for rapid, field-level botanical diagnostics. Powered by a custom-trained ResNet-50 Convolutional Neural Network, the system can accurately identify 38 distinct plant pathogens and healthy states across 14 different crop species.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nitink-22-plant-doctor-ai-app-brvouq.streamlit.app/)

[Launch Plant Doctor AI Web App](https://nitink-22-plant-doctor-ai-app-brvouq.streamlit.app/)

---

## Core Features

* Deep Learning Diagnostics: Utilizes a fine-tuned ResNet-50 model to classify images into 38 distinct categories with high confidence.
* Auto-Isolate Leaf (Alpha Matting): Integrates rembg with advanced alpha matting to automatically strip complex backgrounds, allowing the AI to focus purely on the botanical structure.
* AI Focus X-Ray (Grad-CAM): Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps in real-time, overlaying a visual representation of the exact cellular structures that triggered the AI's diagnosis.
* Actionable Protocols: Maps identified pathogens directly to an internal database of recommended agricultural treatments and mitigation strategies.
* Dual-Input Processing: Supports both local file uploads and live device-camera capture streams.
* Enterprise UI/UX: Features a responsive, dark-mode CSS dashboard optimized for both desktop analysis and mobile field-use.

---

## Evaluation Metrics

* **Validation Accuracy:** 98.09%
* **F1-Score:** ~0.98 (Macro/Weighted)
* **Classification Report:** 
```text
                                                    precision    recall  f1-score   support

                                Apple___Apple_scab       0.99      0.99      0.99       372
                                 Apple___Black_rot       1.00      0.99      1.00       402
                          Apple___Cedar_apple_rust       0.99      1.00      1.00       359
                                   Apple___healthy       1.00      0.99      0.99       416
                               Blueberry___healthy       0.99      1.00      0.99       391
          Cherry_(including_sour)___Powdery_mildew       1.00      0.99      1.00       355
                 Cherry_(including_sour)___healthy       1.00      0.99      0.99       372
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot       0.97      0.94      0.95       297
                       Corn_(maize)___Common_rust_       1.00      1.00      1.00       379
               Corn_(maize)___Northern_Leaf_Blight       0.95      0.97      0.96       401
                            Corn_(maize)___healthy       1.00      1.00      1.00       368
                                 Grape___Black_rot       0.98      0.99      0.98       390
                      Grape___Esca_(Black_Measles)       0.99      0.98      0.98       377
        Grape___Leaf_blight_(Isariopsis_Leaf_Spot)       1.00      1.00      1.00       330
                                   Grape___healthy       0.99      1.00      1.00       341
          Orange___Haunglongbing_(Citrus_greening)       0.99      1.00      1.00       395
                            Peach___Bacterial_spot       0.98      1.00      0.99       343
                                   Peach___healthy       1.00      0.99      1.00       332
                     Pepper,_bell___Bacterial_spot       1.00      0.98      0.99       407
                            Pepper,_bell___healthy       0.98      0.99      0.99       357
                             Potato___Early_blight       1.00      0.99      0.99       400
                              Potato___Late_blight       0.96      0.99      0.98       369
                                  Potato___healthy       1.00      0.98      0.99       368
                               Raspberry___healthy       1.00      0.99      0.99       355
                                 Soybean___healthy       0.99      1.00      0.99       430
                           Squash___Powdery_mildew       1.00      0.99      1.00       335
                          Strawberry___Leaf_scorch       1.00      1.00      1.00       348
                              Strawberry___healthy       0.99      1.00      1.00       353
                           Tomato___Bacterial_spot       0.98      0.95      0.97       335
                             Tomato___Early_blight       0.91      0.92      0.91       398
                              Tomato___Late_blight       0.97      0.92      0.94       365
                                Tomato___Leaf_Mold       0.97      0.97      0.97       392
                       Tomato___Septoria_leaf_spot       0.95      0.94      0.95       362
     Tomato___Spider_mites Two-spotted_spider_mite       0.94      0.92      0.93       355
                              Tomato___Target_Spot       0.87      0.92      0.89       371
            Tomato___Tomato_Yellow_Leaf_Curl_Virus       0.98      0.99      0.99       374
                      Tomato___Tomato_mosaic_virus       0.99      1.00      0.99       397
                                  Tomato___healthy       0.98      0.98      0.98       368

                                          accuracy                           0.98     14059
                                         macro avg       0.98      0.98      0.98     14059
                                      weighted avg       0.98      0.98      0.98     14059
```

---

## Supported Crops
The model is currently trained to process leaves from the following 14 species:
Apple | Blueberry | Cherry | Corn | Grape | Orange | Peach | Pepper (Bell) | Potato | Raspberry | Soybean | Squash | Strawberry | Tomato

---

## Local Installation & Setup

If you wish to run Plant Doctor AI locally on your own machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/nitink-22/Plant_Doctor_AI.git
cd Plant_Doctor_AI
```

### 2. Create and activate a virtual environment

**Windows:**
```powershell
python -m venv env
.\env\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the Model Weights
Ensure your trained weights file (`enterprise_plant_model.pth`) is placed directly in the root directory of the project.

### 5. Launch the Application
```bash
streamlit run app.py
```

---

## Project Structure

```text
Plant_Doctor_AI/
├── app.py
├── code_file.ipynb
├── enterprise_plant_model.pth
├── requirements.txt
├── .gitignore
└── README.md
```

---
## Technology Stack
* Frontend/Deployment: Streamlit
* Deep Learning Framework: PyTorch
* Computer Vision: OpenCV
* Background Removal: Rembg
* Data Processing: NumPy & Pillow

---

## Author

**Nitin Kumar**

*Machine Learning Student*




