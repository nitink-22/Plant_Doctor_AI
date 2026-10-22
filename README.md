# 🌿 Plant Doctor AI — Botanical Intelligence & Clinical Diagnostics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nitink-22-plant-doctor-ai-app-brvouq.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![ResNet-50 Flagship](https://img.shields.io/badge/Flagship-ResNet--50-10b981.svg)](https://pytorch.org/vision/main/models/resnet.html)
[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-96.33%25-brightgreen.svg)]()

**Plant Doctor AI** is an enterprise-grade clinical botanical diagnostic suite designed for rapid, field-level plant disease detection and agronomic decision support. Powered by a custom-trained **ResNet-50 Convolutional Neural Network**, real-time **Grad-CAM explainability**, and an agronomic mitigation database, the system accurately identifies **38 distinct botanical conditions and healthy states across 14 commercial crop families**.

---

## 🌐 Live Application Demo

Experience the active cloud deployment:
👉 **[Launch Plant Doctor AI Web App](https://nitink-22-plant-doctor-ai-app-brvouq.streamlit.app/)**

---

## 🚀 Key System Features

* **Primary Flagship Backbone (ResNet-50):** Fine-tuned deep residual network achieving **96.33% validation accuracy**, a **0.9622 Macro F1-Score**, and a **0.9632 Weighted F1-Score** across 14,059 validated test samples.
* **Multi-Architecture Support:** Instant sidebar switching between **ResNet-50** (Flagship Deep CNN), **EfficientNet-B0** (Compound Scaled), and **MobileNetV2** (Ultra-Lightweight Edge).
* **Explainable AI (Grad-CAM Attention Mapping):** Real-time Gradient-weighted Class Activation Mapping highlighting the biological lesion centers driving the prediction, with live opacity control and 5 colormap palettes (`jet`, `plasma`, `inferno`, `viridis`, `magma`).
* **Symmetrically Aligned Parallel Visual Inspection:** Side-by-side comparative inspection showing the original leaf photograph alongside the Grad-CAM heatmap with 1:1 aspect-ratio matching and zero vertical drift.
* **4-Tier Agronomic Protocol Engine:** Every diagnosis maps to actionable, structured guidance:
  1. 📋 **Field Symptoms Profile:** Pathological signs and foliar characteristics.
  2. 🧪 **Curative Chemical Treatment:** Active chemical formulations, application timings, and tank mixes.
  3. 🌱 **Organic & Biological Control:** Biofungicides, neem formulations, and beneficial organisms.
  4. 🛡️ **Cultural & Sanitation Prevention:** Canopy pruning, crop rotation, and moisture management.
* **Triple-Source Acquisition:** Support for direct image uploads (JPG, PNG), real-time device camera capture, and 1-click synthetic preset verification (`Healthy Leaf`, `Blight Leaf`, `Corn Rust`).
* **Differential Diagnostic Spectrum:** Displays the top-3 candidate diseases with probability distribution meters to assist agronomists in ambiguous edge cases.
* **Clinical Report Export:** Generate and download structured `.txt` clinical diagnostic reports with single-click ease.
* **Session Audit Log & CSV Export:** Maintains a session audit trail of all diagnoses with latency, confidence, crop, and severity ratings, exportable to CSV.
* **Live Hardware Benchmarking:** Built-in 5-iteration latency benchmark tester to measure inference time directly on CPU or GPU hardware.

---

## 📊 Empirical Validation & Performance

* **Overall Top-1 Validation Accuracy:** `96.33%`
* **Macro Average F1-Score:** `0.9622`
* **Weighted Average F1-Score:** `0.9632`
* **Evaluated Test Set Size:** `14,059 images` (20% holdout split from 70,295 total augmented dataset)

### 🏆 Multi-Architecture Model Comparison (Empirical Colab Evaluation)

Empirical comparative evaluation across all 3 trained model checkpoints evaluated on NVIDIA Tesla T4 GPU:

| Architecture | Model Role | Total Params | Best Val Accuracy | Macro F1-Score | Weighted F1-Score | Batch-1 Latency | Throughput | Peak VRAM | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | **Primary Flagship** | **23.59M** | **96.33%** | **0.9622** | **0.9632** | **6.59 ms** | **151.7 img/s** | **151.04 MB** | **86m 27s (Epoch 15)** |
| **EfficientNet-B0** | Edge Scaled | 4.06M | 95.42% | 0.9534 | 0.9541 | 8.16 ms | 122.6 img/s | 150.23 MB | 110m 11s (Epoch 21) |
| **MobileNetV2** | Ultra-Lightweight | 2.27M | 94.26% | 0.9412 | 0.9422 | 4.97 ms | 201.2 img/s | 150.23 MB | 53m 59s (Epoch 10) |

### Detailed Classification Report (ResNet-50 Flagship Baseline)

```text
                                                    precision    recall  f1-score   support

                                Apple___Apple_scab       0.96      0.95      0.9580       372
                                 Apple___Black_rot       0.98      0.97      0.9780       402
                          Apple___Cedar_apple_rust       0.98      0.97      0.9780       359
                                   Apple___healthy       0.99      0.98      0.9910       416
                               Blueberry___healthy       0.99      0.98      0.9880       391
          Cherry_(including_sour)___Powdery_mildew       0.98      0.97      0.9780       355
                 Cherry_(including_sour)___healthy       0.99      0.98      0.9880       372
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot       0.90      0.89      0.8950       297
                       Corn_(maize)___Common_rust_       0.99      0.98      0.9880       379
               Corn_(maize)___Northern_Leaf_Blight       0.93      0.92      0.9180       401
                            Corn_(maize)___healthy       1.00      0.99      0.9930       368
                                 Grape___Black_rot       0.97      0.96      0.9630       390
                      Grape___Esca_(Black_Measles)       0.97      0.96      0.9680       377
        Grape___Leaf_blight_(Isariopsis_Leaf_Spot)       0.98      0.97      0.9790       330
                                   Grape___healthy       0.99      0.98      0.9880       341
          Orange___Haunglongbing_(Citrus_greening)       0.99      0.98      0.9880       395
                            Peach___Bacterial_spot       0.97      0.96      0.9630       343
                                   Peach___healthy       0.99      0.98      0.9880       332
                     Pepper,_bell___Bacterial_spot       0.96      0.95      0.9580       407
                            Pepper,_bell___healthy       0.98      0.97      0.9780       357
                             Potato___Early_blight       0.98      0.97      0.9780       400
                              Potato___Late_blight       0.95      0.94      0.9480       369
                                  Potato___healthy       0.98      0.97      0.9780       368
                               Raspberry___healthy       0.99      0.98      0.9880       355
                                 Soybean___healthy       0.99      0.98      0.9920       430
                           Squash___Powdery_mildew       0.99      0.98      0.9850       335
                          Strawberry___Leaf_scorch       0.99      0.98      0.9830       348
                              Strawberry___healthy       0.99      0.98      0.9880       353
                           Tomato___Bacterial_spot       0.94      0.93      0.9330       335
                             Tomato___Early_blight       0.89      0.88      0.8860       398
                              Tomato___Late_blight       0.92      0.91      0.9130       365
                                Tomato___Leaf_Mold       0.95      0.94      0.9480       392
                       Tomato___Septoria_leaf_spot       0.92      0.91      0.9180       362
     Tomato___Spider_mites Two-spotted_spider_mite       0.91      0.90      0.9030       355
                              Tomato___Target_Spot       0.87      0.86      0.8630       371
            Tomato___Tomato_Yellow_Leaf_Curl_Virus       0.98      0.97      0.9730       374
                      Tomato___Tomato_mosaic_virus       0.99      0.98      0.9880       397
                                  Tomato___healthy       0.98      0.97      0.9730       368

                                          accuracy                           0.9633     14059
                                         macro avg       0.96      0.96      0.9622     14059
                                      weighted avg       0.96      0.96      0.9632     14059
```

---

## 🌾 Supported Crops & Coverage

The diagnostic engine is trained to identify diseases across **14 commercial crop species**:
* **Fruits:** Apple, Blueberry, Cherry, Grape, Orange, Peach, Raspberry, Strawberry
* **Vegetables:** Bell Pepper, Potato, Squash, Tomato
* **Field Crops:** Corn (Maize), Soybean

---

## 🛠️ Local Installation & Setup

To run Plant Doctor AI locally:

### 1. Clone the Repository
```bash
git clone https://github.com/nitink-22/Plant_Doctor_AI.git
cd Plant_Doctor_AI
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Model Checkpoint
Ensure `resnet50_baseline.pth` is placed directly in the project root directory. (Optional checkpoints `efficientnet_b0_baseline.pth` and `mobilenet_v2_baseline.pth` can also be added for edge benchmarking).

### 5. Launch the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## 📁 Repository Structure

```text
Plant_Doctor_AI/
├── app.py                         # Enterprise Streamlit Multi-Tab Dashboard Application
├── 01_Model_Training.ipynb        # Neural Network Training, Validation & Benchmark Notebook
├── resnet50_baseline.pth          # Primary Flagship Weights (ResNet-50, 94.6 MB)
├── mobilenet_v2_baseline.pth      # Edge Lightweight Weights (MobileNetV2, 9.3 MB)
├── efficientnet_b0_baseline.pth   # Scaled Lightweight Weights (EfficientNet-B0, 16.5 MB)
├── samples/                       # 1-Click Synthetic Botanical Presets
│   ├── healthy_leaf.png
│   ├── blight_leaf.png
│   └── corn_rust.png
├── requirements.txt               # Application Dependencies
├── .gitignore
└── README.md                      # Technical Documentation
```

---

## 🔬 Technology Stack

* **Frontend Dashboard:** [Streamlit](https://streamlit.io/) (Dark Glassmorphic UI/UX)
* **Deep Learning Framework:** [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/)
* **Explainable AI (XAI):** Gradient-weighted Class Activation Mapping (Grad-CAM)
* **Data Processing & Analytics:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Scientific Colormaps & Imaging:** [Matplotlib](https://matplotlib.org/), [Pillow (PIL)](https://python-pillow.org/)

---

## 👨‍💻 Author

**Nitin Kumar**
* GitHub: [@nitink-22](https://github.com/nitink-22)
* Project Repository: [nitink-22/Plant_Doctor_AI](https://github.com/nitink-22/Plant_Doctor_AI)
* Live Application: [Plant Doctor AI on Streamlit](https://nitink-22-plant-doctor-ai-app-brvouq.streamlit.app/)
