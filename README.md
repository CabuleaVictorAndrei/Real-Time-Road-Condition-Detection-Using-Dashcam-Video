# Real-Time Road Condition Classification on Edge Devices

## Overview
This project presents a hardware-aware deployment framework for real-time road surface classification using dashcam video. The system is designed to operate efficiently on resource-constrained edge devices by combining deep learning with model optimization techniques such as INT8 quantization and hardware acceleration.

The pipeline enables accurate classification of road conditions (*clear*, *wet*, *snowy*) while maintaining real-time performance suitable for automotive applications.

---

## Key Features
- End-to-end edge deployment framework
- EfficientNet-B3 backbone with a lightweight classification head
- INT8 quantization for integer-only inference
- Hardware-aware model conversion (PyTorch → ONNX → TensorFlow → TFLite)
- Real-time inference with multi-threaded video processing
- Optimized for low-latency execution on embedded systems

---

### Dataset Details

- **Total images:** 17,579 RGB images  
- **Classes:**  
  - `clear` — dry road surfaces under normal conditions  
  - `wet` — damp or wet roads with visible moisture, reflections, or puddles  
  - `snowy` — partially or fully snow-covered or icy road surfaces  

### Class Distribution

| Class   | Train | Validation | Total |
|--------|------:|-----------:|------:|
| Clear (Dry) | 4590 | 1177 | 5767 |
| Wet (Damp)  | 4740 | 1126 | 5866 |
| Snowy (Icy/Snow) | 4732 | 1214 | 5946 |
| **Total** | **14062** | **3517** | **17579** |

### Notes
- The dataset is **approximately balanced** across all classes.
- Images were extracted from **real-world dashcam videos**, ensuring diverse conditions.
- Each sample was **manually annotated and verified** to maintain labeling quality.
- Ambiguous or low-quality samples were removed to ensure **clear class separation**.
- The dataset reflects realistic driving scenarios, including:
  - varying illumination  
  - reflections and glare  
  - partial snow coverage  
  - mixed surface conditions  

### Data Split
- **Training set:** 14,062 images (~80%)  
- **Validation set:** 3,517 images (~20%)  

This distribution supports stable supervised learning while remaining representative of real-world conditions.

### Processing:
- Manual annotation and validation  
- Removal of ambiguous samples  
- Cropping to focus on road regions  
- 80/20 train-validation split  
- Near-balanced class distribution  

### Augmentation:
- Horizontal flipping  
- Rotation (±10°)  
- Translation (±5%)  
- Brightness/contrast/saturation adjustments  

---

## Framework Pipeline
The proposed framework consists of the following stages:

1. **Dataset Preparation**  
2. **Model Training (EfficientNet-B3)**  
3. **Model Conversion & INT8 Quantization**  
4. **Edge Deployment & Real-Time Inference**

---

## Model Details
- **Architecture:** EfficientNet-B3  
- **Input size:** 128 × 128  
- **Optimizer:** Adam  
- **Loss function:** Cross-entropy  
- **Training strategy:** Transfer learning with ImageNet weights  

---

## Deployment
- Converted to TensorFlow Lite (TFLite)
- Fully INT8 quantized (weights + activations)
- Optimized for Neural Processing Unit (NPU)
- Multi-threaded pipeline for real-time video processing

---

## Performance
- **Accuracy:** up to ~99% (FP32), ~98.8% (INT8)  
- **Real-time inference:** ~35 FPS (end-to-end)  
- **Memory reduction:** ~4× compared to FP32  

---

## Applications
- Driver assistance systems  
- Autonomous driving perception  
- Road condition monitoring  
- Edge AI vision systems  

---

## Repository Structure
```bash
├── train/              # Training dataset (organized by class)
├── validation/         # Validation dataset (organized by class)
├── src/                # Source code
│   ├── preprocessing/  # Image cropping and preprocessing scripts
│   ├── training/       # Model training scripts (PyTorch)
│   ├── inference/      # Real-time inference pipeline
│   └── conversion/     # Model conversion and quantization pipeline
└── README.md
