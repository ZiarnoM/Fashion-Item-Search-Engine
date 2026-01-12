"""
Professional dataset visualization for Fashion-MNIST
Creates publication-quality figures for report
"""

import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
import json

# Fashion-MNIST class names
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Set beautiful style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


class RGBWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img, label


def create_class_grid(n_per_class=10):
    """Create grid showing examples from each class"""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataset = RGBWrapper(dataset)
    
    # Collect examples per class
    class_examples = {i: [] for i in range(10)}
    
    for img, label in dataset:
        if len(class_examples[label]) < n_per_class:
            class_examples[label].append(img)
        if all(len(examples) == n_per_class for examples in class_examples.values()):
            break
    
    # Create figure
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class*1.5, 15))
    
    for class_idx in range(10):
        for img_idx in range(n_per_class):
            ax = axes[class_idx, img_idx]
            img = class_examples[class_idx][img_idx]
            
            # Convert to displayable format
            img_display = img.permute(1, 2, 0).numpy()
            ax.imshow(img_display, cmap='gray')
            ax.axis('off')
            
            # Add class name on first column
            if img_idx == 0:
                ax.text(-0.1, 0.5, CLASS_NAMES[class_idx], 
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       rotation=90, va='center', ha='right')
    
    plt.suptitle('Fashion-MNIST Dataset: Sample Images by Class', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/dataset_class_examples.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dataset_class_examples.png")
    plt.close()


def create_class_distribution():
    """Create beautiful bar chart of class distribution"""
    
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True)
    
    train_labels = [label for _, label in train_data]
    test_labels = [label for _, label in test_data]
    
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training set
    x = list(train_counts.keys())
    y = [train_counts[i] for i in x]
    bars1 = ax1.bar(x, y, color=sns.color_palette("husl", 10), edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([CLASS_NAMES[i] for i in x], rotation=45, ha='right')
    ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax1.set_title('Training Set Distribution (60,000 images)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)
    
    # Test set
    x = list(test_counts.keys())
    y = [test_counts[i] for i in x]
    bars2 = ax2.bar(x, y, color=sns.color_palette("husl", 10), edgecolor='black', linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([CLASS_NAMES[i] for i in x], rotation=45, ha='right')
    ax2.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax2.set_title('Test Set Distribution (10,000 images)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Fashion-MNIST Class Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/dataset_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dataset_distribution.png")
    plt.close()


def create_image_statistics():
    """Analyze and visualize image statistics"""
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    # Sample images for analysis
    sample_size = 1000
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    pixel_means = []
    pixel_stds = []
    
    for idx in tqdm(indices, desc="Computing statistics"):
        img, _ = dataset[idx]
        pixel_means.append(img.mean().item())
        pixel_stds.append(img.std().item())
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of mean pixel values
    ax1.hist(pixel_means, bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(pixel_means), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(pixel_means):.3f}')
    ax1.set_xlabel('Mean Pixel Value', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Image Mean Pixel Values', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Histogram of pixel value std
    ax2.hist(pixel_stds, bins=50, color='#FF6B6B', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(pixel_stds), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(pixel_stds):.3f}')
    ax2.set_xlabel('Pixel Value Std Dev', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Image Std Deviation', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/dataset_statistics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dataset_statistics.png")
    plt.close()
    
    # Save stats to JSON
    stats = {
        'mean_pixel_value': {
            'mean': float(np.mean(pixel_means)),
            'std': float(np.std(pixel_means)),
            'min': float(np.min(pixel_means)),
            'max': float(np.max(pixel_means))
        },
        'std_pixel_value': {
            'mean': float(np.mean(pixel_stds)),
            'std': float(np.std(pixel_stds)),
            'min': float(np.min(pixel_stds)),
            'max': float(np.max(pixel_stds))
        }
    }
    
    with open('results/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("✓ Saved dataset_stats.json")


def create_augmentation_comparison():
    """Show original vs augmented images"""
    
    # Original transform
    original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Augmented transform (same as training)
    aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
    
    # Get 5 random images
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    fig, axes = plt.subplots(5, 3, figsize=(10, 14))
    
    for row, idx in enumerate(indices):
        img_pil, label = dataset[idx]
        
        # Original
        img_original = original_transform(img_pil)
        
        # Augmented versions
        img_aug1 = aug_transform(img_pil)
        img_aug2 = aug_transform(img_pil)
        
        # Plot
        for col, (img, title) in enumerate([
            (img_original, 'Original'),
            (img_aug1, 'Augmented 1'),
            (img_aug2, 'Augmented 2')
        ]):
            ax = axes[row, col]
            # Convert to displayable
            if img.shape[0] == 1:
                img = img.squeeze(0)
                ax.imshow(img.numpy(), cmap='gray')
            else:
                ax.imshow(img.permute(1, 2, 0).numpy())
            
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[label], fontsize=10, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/dataset_augmentation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dataset_augmentation.png")
    plt.close()


def create_confusion_classes():
    """Show similar/confusing classes side by side"""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataset = RGBWrapper(dataset)
    
    # Define confusing class pairs
    confusing_pairs = [
        (0, 6),  # T-shirt vs Shirt
        (2, 4),  # Pullover vs Coat
        (5, 7),  # Sandal vs Sneaker
        (7, 9),  # Sneaker vs Ankle boot
    ]
    
    fig, axes = plt.subplots(len(confusing_pairs), 10, figsize=(18, 8))
    
    for pair_idx, (class1, class2) in enumerate(confusing_pairs):
        # Collect examples
        class1_examples = []
        class2_examples = []
        
        for img, label in dataset:
            if label == class1 and len(class1_examples) < 5:
                class1_examples.append(img)
            if label == class2 and len(class2_examples) < 5:
                class2_examples.append(img)
            if len(class1_examples) == 5 and len(class2_examples) == 5:
                break
        
        # Plot
        for i in range(5):
            # Class 1
            ax = axes[pair_idx, i]
            img = class1_examples[i].permute(1, 2, 0).numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(CLASS_NAMES[class1], fontsize=10, fontweight='bold', color='blue')
            
            # Class 2
            ax = axes[pair_idx, i + 5]
            img = class2_examples[i].permute(1, 2, 0).numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(CLASS_NAMES[class2], fontsize=10, fontweight='bold', color='red')
    
    plt.suptitle('Similar/Confusing Fashion Classes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/dataset_confusing_classes.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dataset_confusing_classes.png")
    plt.close()


def create_dataset_summary_infographic():
    """Create beautiful summary infographic"""
    
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True)
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Fashion-MNIST Dataset Summary', fontsize=20, fontweight='bold', y=0.98)
    
    # Dataset info (top left)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    info_text = f"""
    **Fashion-MNIST Dataset Overview**
    
    • Total Images: {len(train_data) + len(test_data):,} (Training: {len(train_data):,} | Test: {len(test_data):,})
    • Image Size: 28×28 pixels (grayscale) → Resized to 224×224 for model
    • Classes: 10 fashion categories
    • Balanced: ~6,000 images per class in training set
    • Source: Zalando Research (2017)
    • Purpose: Drop-in replacement for MNIST for benchmarking
    """
    
    ax1.text(0.5, 0.5, info_text, transform=ax1.transAxes, 
            fontsize=11, va='center', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Sample images (middle)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis('off')
    
    # Get sample images
    transform = transforms.ToTensor()
    sample_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    n_samples = 10
    sample_images = []
    sample_labels = []
    for i in range(n_samples):
        idx = np.random.randint(0, len(sample_dataset))
        img, label = sample_dataset[idx]
        sample_images.append(img)
        sample_labels.append(label)
    
    # Create mini grid
    mini_fig = plt.figure(figsize=(12, 1.5))
    for i in range(n_samples):
        ax = mini_fig.add_subplot(1, n_samples, i + 1)
        ax.imshow(sample_images[i].squeeze(), cmap='gray')
        ax.set_title(CLASS_NAMES[sample_labels[i]], fontsize=8)
        ax.axis('off')
    
    mini_fig.savefig('results/temp_samples.png', dpi=150, bbox_inches='tight')
    plt.close(mini_fig)
    
    # Load and display in main figure
    sample_img = plt.imread('results/temp_samples.png')
    ax2.imshow(sample_img)
    ax2.set_title('Sample Images from Dataset', fontsize=12, fontweight='bold')
    
    # Stats (bottom)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')
    stats_text = """
    **Training Set:**
    • Images: 60,000
    • Split: 80% train, 20% val
    • Augmentation: Yes
    """
    ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, fontsize=10, va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    test_text = """
    **Test Set:**
    • Images: 10,000
    • Usage: Final evaluation
    • No augmentation
    """
    ax4.text(0.1, 0.5, test_text, transform=ax4.transAxes, fontsize=10, va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    model_text = """
    **Preprocessing:**
    • Resize: 224×224
    • Convert: Grayscale→RGB
    • Normalize: ImageNet stats
    """
    ax5.text(0.1, 0.5, model_text, transform=ax5.transAxes, fontsize=10, va='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.savefig('results/dataset_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved dataset_summary.png")
    plt.close()
    
    # Clean up temp file
    Path('results/temp_samples.png').unlink(missing_ok=True)


def main():
    """Generate all dataset visualizations"""
    
    print("="*60)
    print("Creating Dataset Visualizations")
    print("="*60 + "\n")
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    print("1. Creating class example grid...")
    create_class_grid(n_per_class=10)
    
    print("\n2. Creating class distribution charts...")
    create_class_distribution()
    
    print("\n3. Computing image statistics...")
    create_image_statistics()
    
    print("\n4. Creating augmentation examples...")
    create_augmentation_comparison()
    
    print("\n5. Creating confusing classes comparison...")
    create_confusion_classes()
    
    print("\n6. Creating dataset summary infographic...")
    create_dataset_summary_infographic()
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("="*60)
    print("\nGenerated files in results/:")
    print("  • dataset_class_examples.png")
    print("  • dataset_distribution.png")
    print("  • dataset_statistics.png")
    print("  • dataset_augmentation.png")
    print("  • dataset_confusing_classes.png")
    print("  • dataset_summary.png")
    print("  • dataset_stats.json")


if __name__ == '__main__':
    main()