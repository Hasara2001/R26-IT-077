# R26-IT-077

Project repository for Group R26-IT-077.

## Group Details
- Group ID: R26-IT-077

## Description
This repository is created for assignment submission and project management.

IT22168122-

IT22222718-

IT22122728- MENDIS S.W.D.V 

IT22177278-

# Multimodal Early Recurrence Prediction of Liver Cancer Using CT Imaging and Clinical Text Analysis




## Cross-Modal Feature Fusion with Attention (CMFA)

This repository contains the implementation of the **Cross-Modal Feature Fusion with Attention (CMFA)** module developed for the research project:

> **“Multimodal Early Recurrence Prediction of Liver Cancer Using CT Imaging and Clinical Text Analysis”**

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

The proposed CMFA architecture introduces:

> **Bidirectional cross-modal attention-based multimodal fusion between CT image features and clinical text features for liver cancer recurrence prediction.**

Unlike traditional baseline fusion methods, the model dynamically learns interactions between modalities before generating fused representations.

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


