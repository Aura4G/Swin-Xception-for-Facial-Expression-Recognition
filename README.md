# Swin-Xception

> A Hybrid model, combining Shifted Window Multi-head Attention with Depthwise Separable Feed-Forward Networks for advanced local feature extraction.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

## Overview

Swin-Xception offers a novel hybrid approach to boosting performance on Facial Expression Recognition (FER) datasets, leveraging the inductive bias
and efficiency-friendly depthwise separable convolutions of Xception architecture in each Feed-Forward Network of a Swin Transformer. This presents
an accuracy boost while avoiding sky-rocketing the parameter count. Swin-Xception also demonstrates excellent competency as a vision transformer backbone
cooperates with SMOTE to refine decision boundaries for minority classes, effectively answering the class imbalance issue many FER datasets are plagued with.

**Architecture diagram**:

![Architecture](https://github.com/Aura4G/Swin-Xception-for-FER/releases/download/SMOTE-Influenced/complete_backbone.png)

---

## Key Features

- **Shifted Window MSA** — brief explanation of why it matters
- **Depthwise Separable FFN** — brief explanation
- **SMOTE-retrained MLP Head** — brief explanation

---

## Requirements

- Python 3.11+
- PyTorch >= 2.9
- CUDA 13.0+ (for GPU training)

Install dependencies:

---

## Installation

```bash
git clone https://github.com/Aura4G/Swin-Xception-for-Facial-Expression-Recognition.git
cd Swin-Xception-for-Facial-Expression-Recognition
```

---

## Usage

### Training

#### End-to-End Training (Stage 1)

```bash
python main.py train\
  --epochs epochs \
```

#### SMOTE Retraining (Stages 2 & 3)

```bash
python main.py smote-retrain\
```

#### Full Training Pipeline (Stages 1, 2 & 3)

```bash
python main.py pipeline\
  --epochs epochs \
```

### Metrics and Insights on a given Swin-Xception Instance

#### Complete Metrics Report

```bash
python main.py metrics\
  --model-path model_path \
```

#### Grad-CAM and Inference on Individual Image

```bash
python main.py gradcam\
  --model-path model_path \
  --img-path img_path \
```

#### Grad-CAM and Inference on Standardised Images

```bash
python main.py gradcam-set\
  --model-path model_path \
```

---

## Live Demo

![Demo](https://github.com/Aura4G/Swin-Xception-for-FER/releases/download/SMOTE-Influenced/demonstration.png)

---

## Model Variants

| Variant | Parameters | Description |
|---------|-----------|-------------|
| `swin_xception_baseline` | 36M | Lightweight, fast inference, local inductive bias with global attention |
| `swin_xception_final` | 36M | Same as Swin-Xception Base, but with Refined Linear Projection Head after training on SMOTE-affected dataset|

---

## Benchmarks

Results on standard benchmarks (dataset name, split, accuracy):

| Model | RAF-DB (host) | FER2013 (holdout) |
|-------|-----------|-----------|
| ResNet50 | 80.38% | 48.91% |
| **Swin-Xception (mine)** | **81.13%** | **46.41%** |
| **Swin-Xception (SMOTE-influenced)** | **79.24** | **45.01%** |

> Hardware: NVIDIA GeForce 5060 RTX. Inference batch size 32. See `src/engine.py`, `src/datasets.py` for reproduction.

---

## Repository Structure

```
Swin-Xception-for-Facial-Expression-Recognition/
├── cam_results/                             # Heatmap output images from image inputs
├── datasets/                                # Store the expected datasets (RAF-DB and FER2013)
├── src/                                     # Source files
│   ├── swinxception.py                      # Main model modules
│   ├── datasets.py                          # Dataset handling and preprocessing
│   ├── engine.py                            # Training/validation functions
│   ├── utils.py                             # Helpers and metrics functions
│   ├── emotion_displays/                    # Folder of customisable images for the live demo
│   └── haar_cascade/                        # Contains independently sourced Haar Cascade Classifier
├── image_figures/                           # Computed confusion matrices and t-SNE clusters
├── notebooks/                               # Development-phase notebooks
│   ├── final_notebook/                      # Final notebook of development directory
│   │   ├── final_experimentation_run.ipynb  # My experimentation notebook
│   │   ├── cam_results/                     # Heatmap output images from image inputs
│   │   └── model_checkpoints/               # Checkpoint Training Progress (State dictionary)
│   ├── resnet50_comparison                  # Dataset handling and preprocessing
│   │   ├── resnet50_comparison.ipynb        # Notebook to train and evaluate ResNet50 baseline
│   │   └── model_checkpoints/               # Checkpoint Training Progress (State dictionary)
│   └── swinsmall_comparison                 # Training/validation functions
│       ├── swinsmall_comparison.ipynb       # Notebook to train and evaluate Swin-Small baseline
│       └── model_checkpoints/               # Checkpoint Training Progress (State dictionary)
├── model_checkpoints/                       # Checkpoint Training Progress (State dictionary)
├── live_demo.py                             # Real-time Facial Expression Classification demo
└── main.py                                  # Control deck for testing all functionality and testing models.
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mlp_ratio` | 6 | Expansion factor for the DS-FFN |
| `third_layer_blocks` | 6 | Number of Swin-Xception blocks on the third layer |
| `dropout` | 0.5 | Dropout rate before Linear Projection |
| `learning_rate` | 1e-4 | Initial learning rate |
| `optimiser` | AdamW | Optimisation algorithm |
| `scheduler` | CosineAnnealingLR | Learning Rate Scheduling algorithm |
| `eta_min` | 1e-6| Minimum learning rate |


---

## Pretrained Weights

Download pretrained checkpoints from [Releases](https://github.com/Aura4G/Swin-Xception-for-Facial-Expression-Recognition/releases)

---

## Citation

Citation is currently pending.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request, especially for large changes.

---

## License

[MIT](LICENSE) © Aura4G
