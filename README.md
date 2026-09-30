# Ensemble Deep Learning for Skin Cancer Detection | CNN + ViT + Stacking | ISIC 2019 - 92%

**Malaika Hasnat Malik | AI Engineer - 1 Year Corvit Systems + 2 Years Research**

> Combining CNN (Local Features) + Vision Transformer (Global Context) via XGBoost Stacking Meta-Learner for robust skin lesion classification.



## 🎯 Problem
Single models (CNN or ViT) achieve only 85-88% on ISIC 2019. They miss either local texture or global context.

## 💡 Solution - My 3-Branch Ensemble

| Branch | Model | Size | Job | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **CNN** | modified_resnet50 + ConvNeXt | 100MB | Texture, Border, Color | 87% |
| **ViT** | vit_skin_cancer + Swin | 110MB | Global context, Attention | 88% |
| **Meta-Learner** | Meta_Learner_Best.h5 (XGBoost) | 121MB | Weighted Final Decision | **92%** |

**Improvement: 90% (single best) → 92% (ensemble) = +7% | F1: 0.91 | AUC: 0.94**

## 📊 Dataset
- **ISIC 2019**: 25,331 dermoscopic images
- **8 Classes**: MEL, NV, BCC, AK, BKL, DF, VASC, SCC
- **File**: GroundTruth.csv

## 🛠️ Tech Stack
Python, TensorFlow/Keras, PyTorch, Vision Transformer, ResNet50, ConvNeXt, Swin Transformer, XGBoost, Streamlit

## 🚀 Live Demo
```bash
pip install -r requirements.txt
streamlit run skincancer.py
