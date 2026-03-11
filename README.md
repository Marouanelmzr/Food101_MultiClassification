# Food-101 Image Classification — From Scratch to Transfer Learning (PyTorch)

A deep learning project focused on **image classification using Convolutional Neural Networks** on the Food-101 dataset.  
The project explores the full deep learning workflow, including data preprocessing, model design, training optimization, and performance analysis.

The objective is to build and evaluate models capable of classifying food images while experimenting with different architectures and training strategies.

---

## Project Overview

Most tutorials hand you a pretrained ResNet. This project does it the hard way first for learning purposes:

**Phase 1** builds a custom CNN from scratch to understand what actually  
drives performance: architecture choices, training dynamics, and  
learning rate scheduling.

**Phase 2** applies transfer learning to all 101 classes, using the  
experimental methodology developed in Phase 1.

---

## Dataset

The project uses the **Food-101 dataset**, a large-scale benchmark dataset for food image classification.

Food-101 contains:

- **101 food categories**
- **1000 images per class**
- **101,000 total images**

Dataset source:

Food-101 Dataset  
https://www.vision.ee.ethz.ch/datasets_extra/food-101/

The dataset is **not included in this repository**.

---

## Phase 1 — Custom CNN

8 targeted experiments, starting from a model that learned nothing (~10%),  
reaching **84.8% test accuracy** in 30 epochs — no pretrained weights.

The main drivers: BatchNorm to stabilise gradient flow, AdaptiveAvgPool2d  
to eliminate the spatial bottleneck, and a two-phase LR schedule  
(LinearLR warmup → CosineAnnealingLR) that outperformed OneCycleLR alone.

All runs tracked on Weights & Biases.

---

## Phase 2 — Transfer Learning *(in progress)*

ResNet-18 backbone, all 101 classes, staged unfreezing with differential  
learning rates. Same experimental methodology as Phase 1.

---

## Stack

PyTorch 2.10 · CUDA 12.8 · torchvision · torch-lr-finder · 
Weights & Biases · RTX 2060 (6 GB)

---

## Installation

Clone the repository:

```
git clone https://github.com/Marouanelmzr/Food101_MultiClassification.git
cd Food101_MultiClassification
```

Install dependencies:

```
pip install -r requirements.txt
```

Download the Food-101 dataset and place it inside a `data/` directory.

---

## Running the Project

Open the main notebook:

```
notebooks/food101_cnn_training.ipynb
```

The notebook contains the full experimental workflow including:

- data preprocessing
- model training
- hyperparameter experiments
- evaluation and visualization

---

## Author

**Marouane Elmozariahi**

Computer Science Student — Deep Learning