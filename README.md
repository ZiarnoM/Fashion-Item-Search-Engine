# Stanford Online Products Search Engine

A visual search engine for finding similar products using deep learning. Built with EfficientNet-B0 and trained using triplet loss for metric learning.

##  Overview

This project implements a **search engine**  that allows users to search for visually similar products from the Stanford Online Products dataset.

### Key Features

- **Visual Similarity Search**: Find similar products based on visual features
- **Deep Learning**: EfficientNet-B0 backbone with custom embedding head
- **Metric Learning**: Trained with triplet loss for semantic embeddings
- **Web Interface**: Interactive Gradio-based UI
- **Real-time**: Fast inference with GPU acceleration

### Performance Metrics

For best model (EfficientNet-B0):
- **Recall@1**: 60.58% (best match is correct 60% of the time)
- **Recall@5**: 82.43% (correct match in top-5, 82% of the time)
- **mAP@10**: 65.44% (mean average precision)
- **Model Size**: 4.7M parameters (lightweight and efficient)

##  Requirements

### Dataset Requirements ✅
- ✅ At least 1000 photos: Stanford dataset has **120,000 images**
- ✅ Evaluation on 10,000+ photos: Test set has **60,502 images**
- ✅ Minimal size 200x200px: Images are resized to 224x224

### Problem Requirements ✅
- ✅ **Search Engine**: 
- ✅ Uses neural networks for feature extraction
- ✅ Implements similarity search in embedding space

### Model Requirements ✅
- ✅ **Pre-trained model on different problem** (transfer learning):
  - Using ImageNet pre-trained EfficientNet-B0
  - Fine-tuned for metric learning with triplet loss
- ✅ **Metric Learning**: +1 point (non-trivial solution)

### Training Requirements ✅
- ✅ Correct loss function: Triplet Loss with semi-hard mining
- ✅ Train/val/test split: 90/10 split + separate test set
- ✅ Performance metrics: Recall@K (K=1,5,10,20), mAP@10
- ✅ Training dynamics: Loss, triplets per batch, learning rate
- ✅ **Data augmentation**: +1 point
  - RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation

### Tools Requirements ✅
- ✅ Git with README: This file
- ✅ **REST API or GUI (Gradio)**:

## 🚀 Quick Start

### 1. Installation

```bash
# Clone your repository
git clone <your-repo-url>
cd <your-repo>

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the Stanford Online Products dataset:

```bash
# Create data directory
mkdir -p data/Stanford_Online_Products

# Download and extract (adjust paths as needed)
# Dataset available at: https://cvgl.stanford.edu/projects/lifted_struct/
# Or from: http://ftp.cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip

cd data
wget http://ftp.cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
unzip Stanford_Online_Products.zip
cd ..
```

Expected structure:
```
data/Stanford_Online_Products/
├── Ebay_train.txt
├── Ebay_test.txt
└── <image_folders>/
```

### 3. Training (Already Done)

You've already trained the models. If you need to retrain:

```bash
# Train EfficientNet-B0 (recommended)
python train.py --dataset stanford --backbone efficientnet_b0 --batch_size 64 --epochs 10 --lr 0.0001

# Or train all models
python train_multiple.py
```

### 4. Run the Demo

```bash
# Make sure your model path is correct in app_stanford.py
# Default: checkpoints/stanford_efficientnet_b0_best.pth

python app_stanford.py
```

The interface will be available at:
- Local: http://localhost:7860
- Public: A shareable link will be displayed in the terminal

##  Model Architecture

```
Input Image (224×224×3)
        ↓
EfficientNet-B0 Backbone
  (Pretrained on ImageNet)
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
Output Embedding (128-dim)
```

**Total Parameters**: 4,729,084
- Backbone: ~4M parameters
- Embedding Head: ~0.7M parameters

## 🎓 Training Details

### Loss Function
**Triplet Loss** with semi-hard negative mining:
```python
loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```
- Margin: 0.5
- Mining strategy: Hardest positive + semi-hard negative

### Hyperparameters
- **Learning rate**: 0.0001 (Adam optimizer)
- **Batch size**: 64
- **Epochs**: 10
- **Weight decay**: 1e-5
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)

### Data Augmentation
Training transforms:
- Resize to 256×256
- RandomResizedCrop(224, scale=(0.8, 1.0))
- RandomHorizontalFlip
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- RandomRotation(15°)
- ImageNet normalization

### Training Time
- EfficientNet-B0: 16 minutes training and 14 minutes evaluation on GPU
- ResNet18: 18 minutes training and 15 minutes evaluation on GPU
- ResNet50: 20 minutes training and 14 minutes evaluation on GPU
- each model trained for 10 epochs.

##  Results

### Model Comparison

| Model | Recall@1 | Recall@5 | Recall@10 | mAP@10 | Parameters |
|-------|----------|----------|-----------|---------|------------|
| **EfficientNet-B0** | **60.58%** | **82.43%** | **88.78%** | **65.44%** | **4.7M** |
| ResNet18 | 40.27% | 70.06% | 81.44% | 47.95% | 11.5M |
| ResNet50 | 12.40% | 42.93% | 63.62% | 23.32% | 24.6M |

**Winner**: EfficientNet-B0 achieves the best performance with the fewest parameters! 🏆

### Why EfficientNet-B0 is Best
1. **Better accuracy**: 60% vs 40% (ResNet18) or 12% (ResNet50)
2. **More efficient**: Only 4.7M params vs 11.5M or 24.6M
3. **Compound scaling**: Balanced depth, width, and resolution
4. **Better convergence**: Lower validation loss and better generalization

### Training Curves
See `results/` directory for detailed training curves showing:
- Training vs validation loss
- Average triplets per batch
- Learning rate schedule

##  How the Search Works

1. **Feature Extraction**: 
   - Input image → EfficientNet backbone → 128-dim embedding
   - Embeddings are L2-normalized (unit vectors)

2. **Similarity Computation**:
   - Cosine similarity: `sim = query_embedding · gallery_embedding`
   - Higher values = more similar products

3. **Ranking**:
   - Sort by similarity scores (descending)
   - Return top-K most similar items

## 📝 Usage Examples

### Command Line (Training)
```bash
# Train with custom parameters
python train.py \
  --dataset stanford \
  --backbone efficientnet_b0 \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.0001 \
  --margin 0.5
```

##  Troubleshooting

### Model not found error
```python
# Update paths in app_stanford.py:
MODEL_PATH = 'checkpoints/stanford_efficientnet_b0_best.pth'  # Your model path
DATA_ROOT = './data/Stanford_Online_Products'  # Your data path
```

### CUDA out of memory
- Reduce gallery size in `app_stanford.py`:
  ```python
  max_gallery_size = 2000  # Reduce from 5000
  ```
- Reduce batch size during embedding extraction:
  ```python
  batch_size = 16  # Reduce from 32
  ```

### Slow inference
- Ensure model is on GPU: Check `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Reduce number of gallery images
- Use smaller top-k values

##  References

### Dataset
- Stanford Online Products Dataset
  - Source: https://cvgl.stanford.edu/projects/lifted_struct/
  - Paper: "Deep Metric Learning via Lifted Structured Feature Embedding" (CVPR 2016)

### Architecture
- EfficientNet: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019)
- PyTorch torchvision models: https://pytorch.org/vision/stable/models.html

### Loss Function
- Triplet Loss: "FaceNet: A Unified Embedding for Face Recognition and Clustering" (CVPR 2015)
- Semi-hard mining: "In Defense of the Triplet Loss for Person Re-Identification" (arXiv 2017)

### Libraries
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