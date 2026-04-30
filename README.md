# Swin-Xception

> A Hybrid model, combining Shifted Window Multi-head Attention with Depthwise Separable Feed-Forward Networks for advanced local feature extraction.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

## Overview

Swin-Xception offers a novel hybrid approach to boosting performance on Facial Expression Recognition (FER) datasets, leveraging the inductive bias
and efficiency-friendly depthwise separable convolutions of Xception architecture in each Feed-Forward Network of a Swin Transformer. This presents
an accuracy boost while avoiding sky-rocketing the parameter count. Swin-Xception also demonstrates excellent competency as a vision transformer backbone
cooperates with SMOTE to refine decision boundaries for minority classes, effectively answering the class imbalance issue many FER datasets are plagued with.

---

## Architecture diagrams

### Swin-Xception Backbone
![Architecture1](https://github.com/Aura4G/Swin-Xception-for-FER/releases/download/SMOTE-Influenced/complete_backbone.png)

### Swin-Xception Block
![Architecture2](https://github.com/Aura4G/Swin-Xception-for-FER/releases/download/SMOTE-Influenced/swinx_block.png)

### Depthwise Separable Feed-Forward Network
![Architecture3](https://github.com/Aura4G/Swin-Xception-for-FER/releases/download/SMOTE-Influenced/ds_ffn.png)

---

## Key Features

- **Shifted Window MSA** — enables cross-window attention between patches by shifting the partition grid each layer, capturing long-range dependencies without the quadratic cost of full self-attention
- **Depthwise Separable FFN** — retains spatially-aware computation in the feed-forward layer, remaining streamlined while introducing an inductive bias to draw greater significance to fine-grained local features
- **SMOTE-retrained MLP Head** — facilitates further learning from minority classes, by freezing and extracting features & labels from the model backbone, applying SMOTE to create a uniform distribution of features
                                 per class, and retraining the model's Linear Projection head on the balanced feature and label set.

---

## Requirements

- Python 3.11+
- PyTorch >= 2.9
- CUDA 13.0+ (for GPU training)

Install dependencies:
- imbalanced-learn
- timm
- cv2
- matplotlib
- scikit-learn
- numpy
- torchvision
- tqdm
- PIL
---

## Installation

```bash
git clone https://github.com/Aura4G/Swin-Xception-for-Facial-Expression-Recognition.git
cd Swin-Xception-for-Facial-Expression-Recognition
```

---

## Usage

### Sourcing datasets

My dataset distributions I utilised are required to observe the reproducible results I achieved:
- RAF-DB: https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset
- FER2013: https://www.kaggle.com/datasets/msambare/fer2013

They are required to be placed in the './datasets' folder located in the root as follows:
```
datasets/
├──RAF-DB/
│  ├──DATASET/
│  │  ├──test/
│  │  │  ├──angry
│  │  │  ├──disgust
│  │  │  ├──fear
│  │  │  ├──happy
│  │  │  ├──neutral
│  │  │  ├──sad
│  │  │  └──surprise
│  │  └──train/
│  │     ├──angry
│  │     ├──disgust
│  │     ├──fear
│  │     ├──happy
│  │     ├──neutral
│  │     ├──sad
│  │     └──surprise
│  ├──test_labels.csv
│  └──train_labels.csv
└──FER2013/
   ├──test/
   │  ├──angry
   │  ├──disgust
   │  ├──fear
   │  ├──happy
   │  ├──neutral
   │  ├──sad
   │  └──surprise
   ├──train/
   │  ├──angry
   │  ├──disgust
   │  ├──fear
   │  ├──happy
   │  ├──neutral
   │  ├──sad
   │  └──surprise
   └──archive.zip
```

### Training

#### End-to-End Training (Stage 1)
Saves a checkpoint to `model_checkpoints/` every epoch, and creates a state dictionary of the E2E trained model after maximum epochs reached.
This achieves `swin_xception_baseline.pth`

```bash
python main.py train\
  --epochs epochs \
```

#### SMOTE Retraining (Stages 2 & 3)
Creates a state dictionary of the complete model with SMOTE Retraining after every epoch in MLP Retraining is complete.
This achieves `swin_xception_final.pth`

```bash
python main.py smote-retrain\
```

#### Full Training Pipeline (Stages 1, 2 & 3)
Runs End-to-End training, subsequent feature extraction and SMOTE, and MLP Head Retraining on SMOTE'd dataset
Running the entire pipeline creates state dictionaries for `swin_xception_baseline.pth` and `swin_xception_final.pth`

```bash
python main.py pipeline\
  --epochs epochs \
```

### Metrics and Insights on a given Swin-Xception Instance

Each of the following scripts requires a complete working model. See [releases](https://github.com/Aura4G/Swin-Xception-for-FER/releases), or complete `train`/`smote-retrain` to
create your own state dictionaries.

#### Complete Metrics Report

Reports:
- Accuracy
- Precision
- Recall
- F1-Score
- WAR
- UAR
- Confusion Matrices
- t-SNE Clusters

```bash
python main.py metrics\
  --model-path model_path \
```

#### Grad-CAM and Inference on Individual Image

Requires a local image for inference.

```bash
python main.py gradcam\
  --model-path model_path \
  --img-path img_path \
```

#### Grad-CAM and Inference on Standardised Images

Runs inference on and produces grad-CAM insights for the first image in each class folder
of RAF-DB

```bash
python main.py gradcam-set\
  --model-path model_path \
```

---

## Live Demo

![Demo](https://github.com/Aura4G/Swin-Xception-for-FER/releases/download/SMOTE-Influenced/demonstration.png)

The live demo application exhibits the predictive potency of Swin-Xception, as well as some practical application, using the PyQt library.
A Haar Cascade classifier detects human faces every frame, draws a bounding box (leftmost widget), and computes inference using the
model to draw SoftMax probabilities of each expression present for that frame (top-right widget). An image display changes depending on the
prominent expression for that frame (bottom-right widget), demonstrating a slight application of the model towards behavioural assessment.

The image displays, located in `emotion_displays/`, can be changed to any preferrable image, but they default to text labels of the expression class
(as seen in the image above). To add/replace images to be displayed in real-time, place your desired image in the `emotion_displays/` folder, and rename
your image to any of the 7 facial expression classes (angry, disgust, fear, happy, neutral, sad, surprise).

This demo requires a complete Swin-Xception state dictionary, either created during training or located in [Releases](https://github.com/Aura4G/Swin-Xception-for-FER/releases).

```bash
python live_demo.py\
```

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
Swin-Xception-for-FER/
├── cam_results/                             # Heatmap output images from image inputs
├── datasets/                                # Store the expected datasets (RAF-DB and FER2013)
├── emotion_displays/                        # Folder of customisable images for the live demo
├── src/                                     # Source files
│   ├── swinxception.py                      # Main model modules
│   ├── datasets.py                          # Dataset handling and preprocessing
│   ├── engine.py                            # Training/validation functions
│   └── utils.py                             # Helpers and metrics functions
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
| `epochs` | 100 | Minimum learning rate |


---

## Released Model State Dictionaries

Download pretrained checkpoints from [Releases](https://github.com/Aura4G/Swin-Xception-for-FER/releases)

---

## Citation

Citation is currently pending.

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request, especially for large changes.

---

## License

[Apache 2.0](LICENSE) © Aria Noroozi
