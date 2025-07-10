# 🧠 Zero-Shot Object Detection with Fast R-CNN and GCN

This project implements a **zero-shot object classification** system that combines:

- **Fast R-CNN** for visual feature extraction,
- **Graph Convolutional Networks (GCN)** for class embedding generation,
- A **Knowledge Graph** (built from ConceptNet and custom semantic relations) that links seen and unseen object classes.

The system enables **inference on unseen classes** without needing any visual examples during training.

---

## 📌 Key Features

- ✅ **Zero-shot capability**: Recognize unseen classes at inference time
- 🧠 **Semantic knowledge integration**: Uses ConceptNet + custom relations
- 🔍 **GCN-based class embedding**: Embeddings reflect semantic context from KB
- 📦 **Modular structure**: Easy to extend, customize, and evaluate

---

## Model flow chart 
<img width="3684" height="5348" alt="zero shot" src="https://github.com/user-attachments/assets/ce193e97-df26-4578-99c3-199c29caa5cb" />

