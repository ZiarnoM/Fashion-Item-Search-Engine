# Visual Search Engine for Stanford Online Products

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-powered visual search engine for finding similar products using metric learning. Built with EfficientNet-B0 and trained with triplet loss on the Stanford Online Products dataset.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Project Requirements](#project-requirements)
- [Technical Details](#technical-details)
- [Citation](#citation)

---

##  Overview

This project implements a **visual similarity search engine** that allows users to search for visually similar products from a large-scale product catalog. Given a query image, the system retrieves and ranks the most similar products based on learned visual features.

### Problem Type: Search Engine
- **Input**: Query product image
- **Output**: Ranked list of visually similar products
- **Application**: E-commerce product discovery, visual recommendation systems

### Technical Approach
The system uses **metric learning** with deep neural networks to:
1. Extract high-dimensional feature embeddings from product images
2. Learn a metric space where visually similar products are close together
3. Perform efficient similarity search using cosine distance

### Key Challenges
- **Large-scale retrieval**: Searching through 60,000+ test images efficiently
- **Fine-grained similarity**: Distinguishing between visually similar but different products
- **Category diversity**: Handling 12 different product categories with varying visual characteristics

---

##  Features

- **Visual Similarity Search**: Find similar products based on visual features
- **Deep Learning**: EfficientNet-B0 backbone with custom embedding head
- **Metric Learning**: Trained with triplet loss for semantic embeddings
- **Web Interface**: Interactive Gradio-based UI
- **Real-time**: Fast inference with GPU acceleration
- **Multiple Architectures**: Compared ResNet18, ResNet50, and EfficientNet-B0

---

##  Performance

### Best Model: EfficientNet-B0

| Metric | Score | Description |
|--------|-------|-------------|
| **Recall@1** | **60.58%** | Best match is correct 60% of the time |
| **Recall@5** | **82.43%** | Correct match in top-5, 82% of the time |
| **Recall@10** | **88.78%** | Correct match in top-10, 89% of the time |
| **mAP@10** | **65.44%** | Mean average precision |
| **Parameters** | **4.7M** | Lightweight and efficient |
| **Model Size** | **55MB** | Compact for deployment |

### Model Comparison

| Model | Recall@1 | Recall@5 | Recall@10 | mAP@10 | Parameters | Size |
|-------|----------|----------|-----------|---------|------------|------|
| **EfficientNet-B0** | **60.58%** | **82.43%** | **88.78%** | **65.44%** | **4.7M** | **55MB** |
| ResNet18 | 40.27% | 70.06% | 81.44% | 47.95% | 11.5M | 132MB |
| ResNet50 | 12.40% | 42.93% | 63.62% | 23.32% | 24.6M | 283MB |

** Winner: EfficientNet-B0** achieves the best performance with the fewest parameters!

### Why EfficientNet-B0 is Best
1.  **Better accuracy**: 60% vs 40% (ResNet18) or 12% (ResNet50)
2.  **More efficient**: Only 4.7M params vs 11.5M or 24.6M
3.  **Compound scaling**: Balanced depth, width, and resolution
4.  **Better convergence**: Lowest validation loss and superior generalization

---

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ZiarnoM/Fashion-Item-Search-Engine.git
cd Fashion-Item-Search-Engine

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the Stanford Online Products dataset:

```bash
# Create data directory
mkdir -p data/Stanford_Online_Products

# Download and extract
cd data
wget http://ftp.cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
unzip Stanford_Online_Products.zip
cd ..
```

**Expected structure:**
```
data/Stanford_Online_Products/
├── Ebay_train.txt
├── Ebay_test.txt
└── <image_folders>/
```

### 3. Run the Demo

```bash
# Launch Gradio interface
python app_stanford.py
```

The interface will be available at:
-  **Local**: http://localhost:7860
-  **Public**: A shareable link will be displayed in the terminal

### 4. Training (Optional)

Pre-trained models are provided, but you can retrain if needed:

```bash
# Train EfficientNet-B0 (recommended)
python train.py --dataset stanford --backbone efficientnet_b0 --batch_size 64 --epochs 10 --lr 0.0001

# Train all models
python train_multiple.py
```

---

##  Dataset

### Stanford Online Products Dataset

**Source**: Stanford Computer Vision Lab  
**Paper**: "Deep Metric Learning via Lifted Structured Feature Embedding" (CVPR 2016)  
**URL**: https://cvgl.stanford.edu/projects/lifted_struct/

### Dataset Statistics

| Split | Images | Products | Categories | Avg. Images/Product |
|-------|--------|----------|------------|---------------------|
| Train | 53,516 | 10,186 | 11,318 | 5.3 |
| Validation | 6,035 | 1,132 | - | 5.2 |
| Test | 60,502 | 11,316 | 12 | 5.3 |
| **Total** | **120,053** | **22,634** | **12** | **5.3** |


### Product Categories (12 classes)

The dataset includes diverse product categories:
-  Bikes
- Cabinets
-  Chairs
-  Coffee makers
-  Fans
- Kettles
-  Lamps
-  Mugs
-  Sofas
- Staplers
-  Tables
- Toasters

### Data Distribution
- **Balanced product representation**: Each product has ~5 images on average
- **No single-sample products**: 0% in all splits, ensuring robust training
- **Train/test split**: Disjoint product sets (no product overlap between train and test)

---

##  Model Architecture

### EfficientNet-B0 Architecture

```
Input Image (224×224×3)
        ↓
EfficientNet-B0 Backbone
  (Pretrained on ImageNet)
  - Compound-scaled mobile architecture
  - Inverted residual blocks
  - Squeeze-excitation modules
        ↓
Global Average Pooling
        ↓
Embedding Head:
  - Linear(1280 → 512)
  - ReLU
  - Dropout(0.2)
  - Linear(512 → 128)
        ↓
L2 Normalization
        ↓
Output Embedding (128-dim unit vector)
```

**Total Parameters**: 4,729,084
- Backbone: ~4.0M parameters
- Embedding Head: ~0.7M parameters

### Transfer Learning

- **Pre-training**: ImageNet (1000-class classification)
- **Fine-tuning**: Metric learning (similarity/retrieval task)
- **Different problem**: Classification → Metric Learning 

### Metric Learning as Non-Trivial Solution 

This implementation uses **triplet loss with semi-hard negative mining**:
1. **Contrastive/Metric Learning**: Learns a metric space where similar items are close
2. **Hard Negative Mining**: Semi-hard mining strategy selects informative triplets dynamically
3. **Research-backed**: Based on FaceNet (CVPR 2015) and recent improvements

---

##  Training

### Loss Function: Triplet Loss with Semi-Hard Mining

**Mathematical Formulation:**
```
L(a, p, n) = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)
```

Where:
- `a` = anchor image
- `p` = positive image (same product as anchor)
- `n` = negative image (different product)
- `f(·)` = embedding function (neural network)
- `margin` = separation margin (0.5)

**Mining Strategy:**
1. **Hard Positive**: Farthest positive sample (same product, maximum distance)
2. **Semi-Hard Negative**: Negatives that are:
   - Farther than the positive (to avoid trivial triplets)
   - Within margin distance (to provide learning signal)
   - If none exist, use hard negative (closest negative)

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | 0.0001 | Balanced convergence for fine-tuning |
| Batch Size | 64 | Maximum size fitting in GPU memory |
| Epochs | 10 | Validation loss plateaued after 8-10 epochs |
| Embedding Size | 128 | Balance between expressiveness and efficiency |
| Margin | 0.5 | Standard value for triplet loss |
| Weight Decay | 1e-5 | Regularization to prevent overfitting |
| Optimizer | Adam | Adaptive learning rates |
| LR Scheduler | ReduceLROnPlateau | Reduce LR by 0.5× when val loss plateaus (patience=3) |

### Data Augmentation

**Training Augmentation:**
```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crops
    transforms.RandomHorizontalFlip(),                    # Mirror flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2,  # Color variations
                          saturation=0.2),
    transforms.RandomRotation(15),                        # Rotation invariance
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],          # ImageNet normalization
                        [0.229, 0.224, 0.225])
])
```

**Validation/Test Augmentation:**
```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),      # Deterministic crop
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])
```

### Training Dynamics Metrics

Tracked metrics during training:
1. **Training & Validation Loss**: Triplet loss convergence
2. **Average Triplets per Batch**: Number of valid triplets found during mining
3. **Learning Rate Schedule**: Adaptive LR changes via ReduceLROnPlateau

### Training Results (EfficientNet-B0)

| Epoch | Train Loss | Val Loss | Train Triplets | Val Triplets | LR |
|-------|------------|----------|----------------|--------------|-----|
| 1 | 0.0998 | 0.4568 | 0.49 | 63.4 | 1e-4 |
| 5 | 0.0797 | 0.4307 | 0.48 | 63.4 | 1e-4 |
| **9** | **0.0794** | **0.4183** | **0.49** | **63.4** | **5e-5** |
| 10 | 0.0766 | 0.4207 | 0.48 | 63.4 | 2.5e-5 |

**Best Validation Loss**: 0.4183 (Epoch 9)

### Training Time

| Model | Training Time | Evaluation Time | Total Time | Time per Epoch |
|-------|--------------|-----------------|------------|----------------|
| **EfficientNet-B0** | **16 min** | **14 min** | **30 min** | **~1.6 min** |
| ResNet18 | 18 min | 15 min | 33 min | ~1.8 min |
| ResNet50 | 20 min | 14 min | 34 min | ~2.0 min |

**Hardware**: NVIDIA RTX 3060 (12GB VRAM)

### Inference Time

- **Single image embedding**: ~5ms on GPU
- **Batch inference (64 images)**: ~80ms on GPU
- **Full gallery search (60K images)**: ~2-3 seconds

---

##  Results

### Evaluation Metrics

Multiple retrieval metrics were used:
1. **Recall@K**: Proportion of queries where correct match appears in top-K results
2. **Mean Average Precision (mAP@K)**: Average precision across all queries
3. **Precision@K**: Average precision at rank K

### Evaluation Protocol

- **Query Set**: All 60,502 test images
- **Gallery Set**: All 60,502 test images
- **Exclusion**: Same product images excluded from results (realistic scenario)
- **Metric**: Cosine similarity in 128-dim embedding space

### Category-Level Retrieval Results

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | mAP@10 | Parameters |
|-------|----------|----------|-----------|-----------|---------|------------|
| **EfficientNet-B0** | **60.58%** | **82.43%** | **88.78%** | **93.40%** | **65.44%** | **4.7M** |
| ResNet18 | 40.27% | 70.06% | 81.44% | 89.23% | 47.95% | 11.5M |
| ResNet50 | 12.40% | 42.93% | 63.62% | 78.45% | 23.32% | 24.6M |

### Precision@K

| Model | P@1 | P@5 | P@10 | P@20 |
|-------|-----|-----|------|------|
| **EfficientNet-B0** | **60.58%** | **50.21%** | **42.37%** | **35.89%** |
| ResNet18 | 40.27% | 38.14% | 34.22% | 30.15% |
| ResNet50 | 12.40% | 20.18% | 24.67% | 27.83% |

### Analysis

**Why EfficientNet-B0 Wins:**
1.  Better accuracy: 60.58% Recall@1 vs 40.27% or 12.40%
2.  More efficient: Only 4.7M parameters (2.4× smaller than ResNet18)
3.  Compound scaling: Balanced depth, width, and resolution
4.  Better features: ImageNet pre-training with compound scaling
5.  Best convergence: Lowest validation loss (0.418) among all models

**ResNet50 Underperformance:**
- Overfitting: More parameters but limited training time (10 epochs)
- Optimization difficulty: Deeper network requires more careful tuning
- Transfer learning mismatch: ResNet50's depth may not align well with this task

### Embedding Space Analysis

EfficientNet-B0 Embeddings:
- **Dimension**: 128
- **Mean**: 0.0017 (well-centered around zero)
- **Std**: 0.0884 (good spread)
- **L2 Norm**: 1.0 (perfectly normalized unit vectors)

---

##  Usage

### Command Line Training

```bash
# Train with custom parameters
python train.py \
  --dataset stanford \
  --backbone efficientnet_b0 \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.0001 \
  --margin 0.5 \
  --embedding_size 128
```

### Web Interface

```bash
# Launch Gradio demo
python app_stanford.py
```

Upload a query image and get top-K similar products instantly!


---

##x Technical Details

### How the Search Works

1. **Feature Extraction**: 
   - Input image → EfficientNet backbone → 128-dim embedding
   - Embeddings are L2-normalized (unit vectors)

2. **Similarity Computation**:
   - Cosine similarity: `sim = query_embedding · gallery_embedding`
   - Higher values = more similar products

3. **Ranking**:
   - Sort by similarity scores (descending)
   - Return top-K most similar items

### Troubleshooting

#### Model not found error
```python
# Update paths in app_stanford.py:
MODEL_PATH = 'checkpoints/stanford_efficientnet_b0_best.pth'  # Your model path
DATA_ROOT = './data/Stanford_Online_Products'  # Your data path
```

#### CUDA out of memory
```python
# Reduce gallery size in app_stanford.py:
max_gallery_size = 2000  # Reduce from 5000

# Or reduce batch size during embedding extraction:
batch_size = 16  # Reduce from 32
```

#### Slow inference
- Ensure model is on GPU: Check `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Reduce number of gallery images
- Use smaller top-k values

---

##  Citation

### Dataset
```bibtex
@inproceedings{song2016deep,
  title={Deep Metric Learning via Lifted Structured Feature Embedding},
  author={Song, Hyun Oh and Xiang, Yu and Jegelka, Stefanie and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4004--4012},
  year={2016}
}
```

### EfficientNet
```bibtex
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={6105--6114},
  year={2019}
}
```

### Triplet Loss
```bibtex
@inproceedings{schroff2015facenet,
  title={Facenet: A unified embedding for face recognition and clustering},
  author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={815--823},
  year={2015}
}
```

---

##  Dependencies

```
torch==2.0+
torchvision==0.15+
gradio==4.0+
pillow==10.0+
numpy==1.24+
pandas==2.0+
matplotlib==3.7+
seaborn==0.12+
scikit-learn==1.3+
tqdm==4.66+
```

See `requirements.txt` for complete list.

---

##  Acknowledgments

- Stanford Computer Vision Lab for the Stanford Online Products dataset
- PyTorch team for the excellent deep learning framework
- Gradio team for the intuitive web interface library

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Michał Żarnowski** (160277)  
🔗 GitHub: [@ZiarnoM](https://github.com/ZiarnoM)

---

** If you find this project helpful, please consider giving it a star!**