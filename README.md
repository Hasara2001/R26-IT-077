# R26-IT-077

Project repository for Group R26-IT-077.

## Group Details
- Group ID: R26-IT-077

## Description
This repository is created for assignment submission and project management.

IT22168122- Wimalasena K.H.N

IT22222718- Weerakoon W.M.U

IT22122728- Mendis S.W.D.V 

IT22177278- Kanishka P.H.A.T

# Multimodal Early Recurrence Prediction of Liver Cancer Using CT Imaging and Clinical Text Analysis

---
## Advanced Image Feature Extraction (AIFE)

This repository contains the implementation of the **Advanced Image Feature Extraction (AIFE)** module developed for the research project:

## 📌 Research Objective

The objective of this component is to automatically extract rich, reproducible, and clinically interpretable image features from raw liver CT scans by:

* Standardising CT scan data across different scanners and hospitals
* Automatically localising and segmenting the liver tumour using 3D UNet
* Selecting the most diagnostically relevant tumour slice
* Extracting and comparing three feature extraction approaches
* Producing a compact fusion-ready feature vector for recurrence prediction

## 🧠 Proposed AIFE Pipeline Architecture

```text
Raw CT Scan (.nii)
        ↓
CT Preprocessing (MONAI)
RAS Orientation · HU Windowing · Resampling · Normalisation
        ↓
3D UNet Segmentation
Tumour Binary Mask Generation
        ↓
Tumour-Guided Slice Selection
Max Tumour-Area Axial Slice
        ↓
┌─────────────────────────────────────────────┐
│         Feature Extraction (3 Models)        │
│                                             │
│  ResNet-50       DenseNet-121    Radiomics   │
│  2048-dim        1024-dim        18-dim ★    │
└─────────────────────────────────────────────┘
        ↓
Best Feature Vector (Radiomics 18-dim)
        ↓
Multimodal Fusion Component (CMFA)
```

---

## 🚀 Key Features

*  UNet-based automatic liver tumour segmentation
* Novel tumour-guided axial slice selection algorithm
* Transfer learning feature extraction using ResNet-50 (2048-dim)
* Transfer learning feature extraction using DenseNet-121 (1024-dim)

 ## 🛠 Technologies Used

* Python 3.11
* PyTorch 2.1.0
* MONAI 1.3.0 (Medical Open Network for AI)
* torchvision 0.16.0
* nibabel 5.1.0
* NumPy 1.26.0
* Pandas 2.1.0
* scikit-learn 1.3.0
* FastAPI 0.115.0

  ## 📂 Dataset

The component uses the **LiTS — Liver Tumour Segmentation Benchmark** dataset:

* Source: Kaggle — andrewmvd/liver-tumor-segmentation
* Format: NIfTI (.nii) 3D CT volumes with segmentation masks

  
## 💡 Novelty

The proposed AIFE pipeline introduces three novel contributions:

> **1.  Radiomic + feature extraction  Pipeline**
> First system to combine IBSI-compliant radiomics with pretrained 2D CNNs and directly compare both approaches on the LiTS dataset specifically for liver recurrence prediction. Most existing works use either pure radiomics or pure CNN — not both with systematic comparison.

> **2. UNet-Guided Tumour Slice Selection**
> Instead of arbitrary mid-slice selection used in most existing 2D pipelines, the algorithm uses the 3D UNet segmentation mask to compute tumour cross-sectional area per axial slice and selects the maximum — guaranteeing features are extracted from the most diagnostically relevant region of the CT scan.

> **3. Fusion-Optimised Compact Feature Vector**
> The 18-dimensional radiomic vector is specifically designed for cross-modal attention fusion with clinical text embeddings — solving the dimensionality compatibility challenge in multimodal medical AI systems.

 
## 🔬 Research Contribution

* Designed and implemented a complete CT image feature extraction pipeline
* Trained a 3D UNet segmentation model achieving validation loss of 0.5577
* Developed a novel tumour-guided slice selection algorithm

  ## 📈 Future Improvements

* Complete full 100-epoch UNet training for improved segmentation accuracy
* Integrate real clinical recurrence labels for validated AUC reporting
* Add GLCM and wavelet radiomic features for richer feature representation
* Implement ClinicalBERT-compatible embedding alignment
* Add batch processing support for multiple CT scans

  ## 📷 Prototype Demonstration

The prototype successfully:

* Accepts NIfTI (.nii) CT scan uploads via web interface
* Runs complete preprocessing pipeline using MONAI
* Performs automatic tumour segmentation using trained 3D UNet
* Selects the most informative tumour slice automatically
* Extracts features using ResNet-50, DenseNet-121, and Radiomics
* Displays all 18 radiomic feature values with clinical labels

 
---
## Contextual Clinical Text Feature Extraction
 type your part....




 



 
---


## Cross-Modal Feature Fusion with Attention (CMFA)

This repository contains the implementation of the **Cross-Modal Feature Fusion with Attention (CMFA)** module developed for the research project:


The CMFA component is responsible for integrating CT-derived imaging features and clinical text features using a bidirectional cross-attention mechanism to generate a unified multimodal representation for downstream recurrence prediction.

---

# 📌 Research Objective

The objective of this component is to improve liver cancer early recurrence prediction by dynamically learning interactions between:

* CT imaging features
* Clinical text features

instead of relying on traditional static fusion methods such as simple concatenation.

---

# 🧠 Proposed CMFA Architecture

```text
Image Features (F_img)
            ↓
     Cross-Attention
            ↓
        Fusion Layer
            ↓
   Fused Representation (F_fused)
            ↓
   Downstream Prediction Module
```

```text
Clinical Text Features (F_text)
            ↓
     Cross-Attention
```

---

# 🚀 Key Features

* Bidirectional Multi-Head Cross-Attention
* Multimodal Feature Fusion
* Dynamic Image-Text Interaction Learning
* 128-Dimensional Fused Feature Representation
* Deep Learning-Based Fusion Architecture
* Validation using Accuracy, F1-Score, and AUC-ROC

---

# 🛠 Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

# 📂 Dataset

The prototype currently uses a structured multimodal liver cancer recurrence dataset containing:

* Tumor morphology features
* Radiomics-inspired imaging features
* Clinical and laboratory information


### Dataset Summary

* Total Patients: 3000+
* Image Feature Dimensions: 12
* Clinical Text Feature Dimensions: 13

---

# ⚙️ Model Architecture

The CMFA model includes:

* Projection Layers
* Bidirectional Multi-Head Cross-Attention
* Fusion Layer
* Lightweight Classification Head (for prototype validation)

---

# 📊 Validation Results

| Metric              | Score  |
| ------------------- | ------ |
| Validation Accuracy | 81%    |
| AUC-ROC             | 0.876  |
| F1-Score            | 0.6275 |

### Output

```text
Fused Feature Shape: [800, 128]
```

The fused multimodal representations are intended for downstream recurrence prediction and explainability modules.

---

# 💡 Novelty

The novelty of the proposed Cross-Modal Feature Fusion with Attention (CMFA) module lies in its ability to dynamically integrate CT imaging features and clinical text features using a bidirectional multi-head cross-attention mechanism for liver cancer early recurrence prediction. Unlike traditional approaches that rely on single-modality analysis or static fusion methods such as simple concatenation, the proposed CMFA architecture enables meaningful interaction learning between imaging and clinical modalities. The model allows CT image embeddings and clinical text embeddings to attend to each other, thereby capturing complex multimodal relationships and clinically relevant patterns that may not be visible when each modality is analyzed independently. Through this attention-guided fusion strategy, the system generates a unified multimodal representation (`F_fused`) that combines structural tumor characteristics and patient-specific clinical context within a shared embedding space. This interaction-aware multimodal fusion approach enhances the model’s ability to support more accurate and context-aware liver cancer recurrence prediction.

---

# 🔬 Research Contribution

* Designed and implemented a CMFA architecture for multimodal fusion
* Developed bidirectional cross-attention between imaging and clinical embeddings
* Generated unified multimodal feature representations
* Evaluated the fusion mechanism using multiple validation metrics

---

# 📈 Future Improvements

* Integration with real CT image embeddings
* ClinicalBERT-based text embeddings
* Deployment via FastAPI / Flask API

---

# 📷 Prototype Demonstration

The prototype successfully:

* Trains the CMFA model
* Generates fused multimodal representations
* Produces recurrence probability predictions
* Visualizes training and validation performance

---

## Predictive Risk Modeling with Explainability

This repository contains the implementation of the **Predictive Risk Modeling with Explainability** component developed for the research project:


This component is responsible for predicting the probability of early liver cancer recurrence using multimodal fused representations generated by the CMFA module. In addition to prediction, the system provides explainable AI outputs to improve transparency and clinical interpretability.

The module combines deep learning–based risk prediction with explainability techniques such as Grad-CAM and SHAP to identify the most influential imaging regions and clinical features contributing to recurrence prediction.

---

# 📌 Research Objective

The objective of this component is to accurately predict early recurrence risk in liver cancer patients while providing interpretable explanations for clinical decision-making.

The system aims to:

* Predict recurrence probability using fused multimodal representations
* Estimate confidence scores alongside predictions
* Highlight important tumor regions in CT scans
* Identify influential clinical text features
* Improve transparency and trust in AI-assisted healthcare systems

---

# 🧠 Proposed Predictive Architecture

```text
Fused Multimodal Features (F_fused)
                    ↓
          Prediction Network
                    ↓
        Recurrence Risk Score
                    ↓
         Confidence Estimation
                    ↓
      Explainability Components
      (Grad-CAM + SHAP/LIME)
                    ↓
     Interpretable AI Predictions
```

---

# 🚀 Key Features

* Deep Learning-Based Risk Prediction
* Multimodal Recurrence Classification
* Confidence Score Estimation
* Explainable AI Integration
* Grad-CAM Visualization for CT Images
* SHAP/LIME-Based Clinical Feature Importance
* Lightweight Neural Prediction Head
* GPU-Compatible Inference Pipeline
* Validation using Accuracy, F1-Score, and AUC-ROC

---

# 🛠 Technologies Used

* Python
* PyTorch
* MONAI
* Captum
* SHAP
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

# 📂 Dataset

The prototype currently uses a multimodal liver cancer recurrence dataset containing:

* CT-derived fused imaging representations
* Clinical text-derived embeddings
* Recurrence labels for supervised prediction

### Dataset Summary

* Total Patients: 3000+
* Fused Feature Dimensions: 128
* Binary Classification:

  * Early Recurrence
  * Non-Recurrence

---

# ⚙️ Model Architecture

The Predictive Risk Modeling module includes:

* Multimodal Prediction Network
* Fully Connected Prediction Layers
* Dropout Regularization
* Confidence Estimation Layer
* Explainability Integration

  * Grad-CAM
  * SHAP / LIME

### Prediction Pipeline

```text
F_fused
   ↓
Dense Layer
   ↓
ReLU + Dropout
   ↓
Dense Layer
   ↓
Sigmoid Output
   ↓
Recurrence Probability
```

---

# 📊 Validation Results

| Metric              | Score |
| ------------------- | ----- |
| Validation Accuracy | 75%   |
| AUC-ROC             | 0.901 |
| F1-Score            | 0.691 |

### Output

```text
Prediction Output Shape: [800, 1]
```

### Example Prediction

```text
Recurrence Probability: 0.87
Confidence Score: 0.91
Risk Category: High Risk
```

The model successfully generates recurrence predictions together with interpretable AI explanations.

---

# 🔍 Explainability Module

The explainability component improves transparency by identifying:

### 🧠 CT Image Explanations

Using **Grad-CAM**, the model highlights:

* Tumor regions
* High-risk imaging areas
* Morphological patterns influencing prediction

### 📄 Clinical Feature Explanations

Using **SHAP/LIME**, the system identifies:

* AFP levels
* Cirrhosis indicators
* Vascular invasion
* Prior treatment history
* Laboratory abnormalities

These outputs help clinicians understand why the model generated a specific prediction.

---

# 💡 Novelty

The proposed Predictive Risk Modeling component introduces:

> **Explainable multimodal recurrence prediction using fused CT imaging and clinical text representations for liver cancer prognosis.**

Unlike traditional black-box prediction systems, this approach provides both:

* Accurate recurrence risk estimation
* Clinically interpretable explanations

---

# 🔬 Research Contribution

* Developed a multimodal recurrence prediction framework
* Integrated explainable AI into medical risk prediction
* Implemented Grad-CAM-based image interpretation
* Implemented SHAP/LIME-based clinical feature interpretation
* Generated interpretable recurrence risk scores
* Evaluated prediction performance using multiple validation metrics

---

# 📈 Future Improvements

* Integration with real-time hospital systems
* Clinical dashboard for visual explanations
* Bayesian uncertainty-aware prediction
* Survival analysis using DeepSurv
* ONNX deployment optimization
* Multi-center clinical validation

---

# 📷 Prototype Demonstration

The prototype successfully:

* Processes fused multimodal representations
* Predicts liver cancer recurrence probability
* Generates confidence-aware predictions
* Produces explainability visualizations
* Highlights influential clinical features
* Visualizes model performance metrics

---

# 🏥 Clinical Impact

This module can support:

* Early recurrence risk assessment
* Personalized treatment planning
* AI-assisted oncology decision support
* Transparent medical AI systems

The explainability-driven design improves clinician trust and supports responsible AI adoption in healthcare.






 
---

