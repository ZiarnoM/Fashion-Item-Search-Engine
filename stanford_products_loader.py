"""
Stanford Online Products Dataset Loader
120K images, 22K unique products, 12 categories, designed for image retrieval

Dataset Structure:
- 12 categories: bicycle, cabinet, chair, coffee_maker, fan, kettle, lamp, mug, sofa, stapler, table, toaster
- 22,634 unique product classes (each product has multiple photos from different angles)
- 120,053 total images
- Task: Learn embeddings where different photos of the SAME product are close together
- Training uses disjoint product classes from testing (different products, not overlapping)

Validation Strategy:
- Split by product classes (not individual samples) since some products have few images
- 90% of products for training, 10% for validation
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from collections import defaultdict

class StanfordProductsDataset(Dataset):
    """
    Stanford Online Products dataset for image retrieval

    The dataset contains multiple photos of the same product from different angles.
    The goal is to learn embeddings where photos of the same product are similar.

    Key IDs in the dataset:
    - class_id (1-22634): Unique product identifier - this is what we train on
    - super_class_id (1-12): Product category (bicycle, chair, etc.)

    Validation Split Strategy:
    Since some products have very few images (2-3), we split by CLASSES rather than samples:
    - 90% of product classes → training set
    - 10% of product classes → validation set
    This ensures each product's images stay together.

    Structure:
    Stanford_Online_Products/
        ├── bicycle_final/
        │   ├── 111085122871_0.JPG
        │   └── ...
        ├── Ebay_train.txt
        └── Ebay_test.txt

    Format of txt files:
    image_id class_id super_class_id path
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to Stanford_Online_Products folder
            split: 'train' or 'test'
            transform: torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Load split file
        if split == 'train':
            split_file = self.root_dir / 'Ebay_train.txt'
        else:
            split_file = self.root_dir / 'Ebay_test.txt'

        # Parse split file
        self.image_paths = []
        self.labels = []
        self.super_labels = []

        print(f"Loading {split} split from {split_file}...")

        with open(split_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    image_id = int(parts[0])
                    class_id = int(parts[1])
                    super_class_id = int(parts[2])
                    path = ' '.join(parts[3:])  # Handle paths with spaces

                    full_path = self.root_dir / path
                    if full_path.exists():
                        self.image_paths.append(str(full_path))
                        self.labels.append(class_id)
                        self.super_labels.append(super_class_id)

        # Create label mapping (original class IDs might not be continuous)
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Remap labels to continuous indices
        self.labels = [self.label_to_idx[label] for label in self.labels]

        print(f"Loaded {len(self.image_paths)} images")
        print(f"Number of classes: {len(unique_labels)}")

        # Create validation split from train if needed
        if split == 'train':
            # Split by classes instead of by samples to avoid stratification issues
            # Some products have very few images, so we split entire classes
            import random
            random.seed(42)

            unique_classes = list(set(self.labels))
            random.shuffle(unique_classes)

            # Use 90% of classes for training, 10% for validation
            n_train_classes = int(0.9 * len(unique_classes))
            train_classes = set(unique_classes[:n_train_classes])
            val_classes = set(unique_classes[n_train_classes:])

            # Split indices based on class membership
            train_indices = [i for i, label in enumerate(self.labels) if label in train_classes]
            val_indices = [i for i, label in enumerate(self.labels) if label in val_classes]

            # Store both splits
            self._all_image_paths = self.image_paths.copy()
            self._all_labels = self.labels.copy()
            self._all_super_labels = self.super_labels.copy()  # Store categories
            self.train_indices = train_indices
            self.val_indices = val_indices

            print(f"Split into {len(train_classes)} train classes and {len(val_classes)} val classes")
            print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

            # Use train indices by default
            self.image_paths = [self._all_image_paths[i] for i in train_indices]
            self.labels = [self._all_labels[i] for i in train_indices]
            self.super_labels = [self._all_super_labels[i] for i in train_indices]

    def set_split(self, split_type):
        """Switch between train/val for training split"""
        if split_type == 'train' and hasattr(self, 'train_indices'):
            self.image_paths = [self._all_image_paths[i] for i in self.train_indices]
            self.labels = [self._all_labels[i] for i in self.train_indices]
            self.super_labels = [self._all_super_labels[i] for i in self.train_indices]
        elif split_type == 'val' and hasattr(self, 'val_indices'):
            self.image_paths = [self._all_image_paths[i] for i in self.val_indices]
            self.labels = [self._all_labels[i] for i in self.val_indices]
            self.super_labels = [self._all_super_labels[i] for i in self.val_indices]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self):
        """Get number of images per class"""
        from collections import Counter
        return Counter(self.labels)


def create_stanford_loaders(root_dir='./data/Stanford_Online_Products', 
                           batch_size=128, 
                           num_workers=4):
    """
    Create train/val/test dataloaders for Stanford Online Products
    """
    
    # Stronger augmentation for this dataset
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = StanfordProductsDataset(root_dir, split='train', transform=train_transform)
    
    # Create validation dataset from train split
    val_dataset = StanfordProductsDataset(root_dir, split='train', transform=test_transform)
    val_dataset.set_split('val')
    
    # Test dataset
    test_dataset = StanfordProductsDataset(root_dir, split='test', transform=test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Important for metric learning
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Save dataset info
    dataset_info = {
        'name': 'Stanford Online Products',
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'num_classes': len(train_dataset.label_to_idx),
        'num_train_classes': len(set(train_dataset.labels)),
        'num_test_classes': len(set(test_dataset.labels)),
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "="*70)
    print("Stanford Online Products Dataset Loaded")
    print("="*70)
    print(f"Training samples:   {len(train_dataset):>8,}")
    print(f"Validation samples: {len(val_dataset):>8,}")
    print(f"Test samples:       {len(test_dataset):>8,}")
    print(f"Total classes:      {len(train_dataset.label_to_idx):>8,}")
    print(f"Batch size:         {batch_size:>8,}")
    print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader


# Quick test
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='./data/Stanford_Online_Products',
                       help='Path to Stanford Online Products dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    print("Testing Stanford Online Products loader...")
    
    try:
        train_loader, val_loader, test_loader = create_stanford_loaders(
            root_dir=args.data_dir,
            batch_size=args.batch_size
        )
        
        # Test loading a batch
        images, labels = next(iter(train_loader))
        print(f"\nFirst batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Unique labels in batch: {len(torch.unique(labels))}")
        
        print("\n✅ Dataset loader working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the dataset is downloaded to the correct path:")
        print(f"  Expected: {args.data_dir}/Ebay_train.txt")
        print(f"  Expected: {args.data_dir}/Ebay_test.txt")
        print("\nDownload from:")
        print("  ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip")