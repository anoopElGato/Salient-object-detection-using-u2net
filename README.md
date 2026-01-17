# Salient Object Detection using U²-Net

This repository implements **Salient Object Detection (SOD)** using a **lightweight U²-Net (U²-Net Lite)** and compares it with a **classical saliency method (Saliency Filters)**.

The project focuses on understanding and comparing **classical computer vision techniques** with **modern deep learning approaches**, while keeping the implementation **simple and easy to follow**.

---

## Objectives

- Implement classical **Saliency Filters** in Python which was originally in C++
- Build a **simplified U²-Net Lite** model  
- Compare classical vs deep learning approaches  
- Analyze performance using quantitative and qualitative metrics  

---

## What is Salient Object Detection?

Salient Object Detection aims to identify the **most visually important object(s)** in an image, similar to how humans naturally focus attention.

**Applications:**
- Image segmentation  
- Object detection  
- Image editing  
- Robotics and vision systems  

---

## Methods Implemented

---

### 1. Saliency Filters (Classical Method)

**File:** `saliency_filters.py`

Based on the paper:  
**Perazzi et al., “Saliency Filters: Contrast Based Filtering for Salient Region Detection”, CVPR 2012**

#### Key Idea
- The image is divided into perceptually homogeneous regions  
- Saliency is computed using:
  - **Color uniqueness**
  - **Spatial distribution**
- No training is required  

#### Characteristics
- Fully unsupervised  
- Mathematical and filter-based  
- Works well on simple, high-contrast images  
- Performs poorly on complex or cluttered scenes  

---

### 2. U²-Net Lite (Deep Learning Method)

U²-Net Lite is a **simplified and lightweight version of U²-Net**, designed to:
- Reduce model size  
- Keep architecture easy to understand  
- Maintain strong segmentation performance  

---

##  U²-Net Lite Architecture

The architecture follows a nested U-structure with **lightweight Residual U-Blocks (RSU-4 Lite)**.

**Key design choices:**
- Reduced encoder–decoder depth  
- Lightweight RSU blocks  
- Multi-scale feature extraction  
- Deep supervision using side outputs  

![U²-Net Lite Architecture](images/Model_Architecture.jpg)

---

##  Loss Function

We use a **combined loss** to improve segmentation quality.


![Loss Functions](images/Loss_functions.png)
We use a combination of Binary Cross-Entropy (BCE) Loss and Dice Loss to balance pixel-level accuracy and object-level shape consistency.
BCE loss helps the model learn correct foreground–background classification for each pixel, while Dice loss focuses on improving overlap between the predicted saliency map and the ground truth.
This combination leads to more accurate boundaries and stable training, especially when foreground and background pixels are highly imbalanced.

---

## Training Details

- **Dataset:** DUTS-TR (train), DUTS-TE (test)  
- **Input size:** 320 × 320  
- **Optimizer:** Adam  
- **Epochs:** 30  
- **Framework:** PyTorch  

---

##  Training Performance

The following metrics were monitored during training:
- Training & validation loss  
- Mean Absolute Error (MAE)  
- F-measure  
- IoU  
- Structure-measure (S-measure)  

![Training Performance](images/training_history.png)

---

## Quantitative Results

### Performance Comparison (DUTS-TE)

| Method | F-measure ↑ | MAE ↓ | Precision | Recall | IoU | S-measure ↑ |
|------|------------|------|----------|--------|-----|------------|
| Saliency Filters | 0.244 | 0.238 | 0.553 | 0.1346 | – | 0.973 |
| **U²-Net Lite (Ours)** | **0.715** | **0.073** | **0.758** | **0.730** | **0.604** | **0.993** |
| U²-Net (Full) | 0.763 | 0.054 | – | – | – | 0.941 |

---

##  Model Efficiency

| Method | Average Time (seconds) |
|------|------------------------|
| Saliency Filters (CPU) | 0.153 |
| U²-Net Lite | 0.120 |
| U²-Net | 0.033 |


---

## Qualitative Results

The visual comparison includes:
- Input image  
- Ground truth  
- U²-Net Lite prediction  
- Saliency Filters output  

U²-Net Lite produces **cleaner object segmentation** and performs better in complex scenes.

![Qualitative Comparison](images/Image_1.png)
![Qualitative Comparison](images/Image_2.jpg)
![Qualitative Comparison](images/Image_4.jpg)
![Qualitative Comparison](images/Image_5.jpg)
![Qualitative Comparison](images/Image_6.png)

---

## Metric Comparison Plots

The plots below compare:
- Structure-measure  
- Precision & Recall  
- Mean Absolute Error  
- F-measure  
- Processing speed  

![Metric Comparison](images/comparison_summary_with_smeasure.png)

---

