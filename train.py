import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from stanford_products_loader import StanfordProductsDataset

# Enable cudnn optimizations
torch.backends.cudnn.benchmark = True


class EmbeddingNet(nn.Module):
    """Neural network that produces embeddings for images"""

    def __init__(self, backbone='resnet18', embedding_size=128, pretrained=True):
        super().__init__()

        # Load backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            num_features = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            num_features = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            num_features = base_model.classifier[1].in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=pretrained)
            num_features = base_model.classifier[1].in_features
            self.backbone = base_model.features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_size)
        )

        self.backbone_name = backbone

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def mine_hard_triplets(embeddings, labels, margin=0.5):
    """
    Mine hard triplets using vectorized operations

    Args:
        embeddings: (batch_size, embedding_dim)
        labels: (batch_size,)
        margin: margin for semi-hard mining
    """
    batch_size = embeddings.size(0)

    # Compute pairwise distances (batch_size, batch_size)
    distances = torch.cdist(embeddings, embeddings, p=2)

    # Create masks
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_not_equal = ~labels_equal

    # Mask out diagonal (distance to self)
    mask_diagonal = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    labels_equal = labels_equal & ~mask_diagonal

    triplets = []

    for i in range(batch_size):
        # Positive: same class, not self
        positive_mask = labels_equal[i]
        if not positive_mask.any():
            continue

        # Negative: different class
        negative_mask = labels_not_equal[i]
        if not negative_mask.any():
            continue

        # Get distances for this anchor
        anchor_positive_dists = distances[i][positive_mask]
        anchor_negative_dists = distances[i][negative_mask]

        # Hard positive: farthest positive
        hardest_positive_idx = torch.argmax(anchor_positive_dists)
        positive_idx = torch.where(positive_mask)[0][hardest_positive_idx]

        # Semi-hard negative: closest negative that's still farther than positive + margin
        hardest_positive_dist = anchor_positive_dists[hardest_positive_idx]

        # Try to find semi-hard negative
        semi_hard_negatives = anchor_negative_dists > hardest_positive_dist
        semi_hard_negatives = semi_hard_negatives & (anchor_negative_dists < hardest_positive_dist + margin)

        if semi_hard_negatives.any():
            # Semi-hard negative exists: pick closest among them
            semi_hard_dists = anchor_negative_dists.clone()
            semi_hard_dists[~semi_hard_negatives] = float('inf')
            negative_idx_in_subset = torch.argmin(semi_hard_dists)
            negative_idx = torch.where(negative_mask)[0][negative_idx_in_subset]
        else:
            # Hard negative: closest negative overall
            hardest_negative_idx = torch.argmin(anchor_negative_dists)
            negative_idx = torch.where(negative_mask)[0][hardest_negative_idx]

        triplets.append((i, positive_idx.item(), negative_idx.item()))

    return triplets


class RGBWrapper(torch.utils.data.Dataset):
    """Wrapper to convert grayscale to RGB"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img, label


def get_data_loaders(dataset_type='fashionmnist', batch_size=128, use_subset=False):
    """Load and prepare data loaders with optimizations"""

    if dataset_type == 'stanford':
        print("Loading Stanford Online Products dataset...")

        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Validation transform (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load datasets
        train_dataset = StanfordProductsDataset(
            root_dir='./data/Stanford_Online_Products',
            split='train',
            transform=train_transform
        )

        val_dataset = StanfordProductsDataset(
            root_dir='./data/Stanford_Online_Products',
            split='train',
            transform=val_transform
        )

        # Split train into train/val
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(train_dataset)))

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        print(f"Train size: {len(train_dataset)}")
        print(f"Val size: {len(val_dataset)}")

    else:  # fashionmnist
        print("Loading Fashion-MNIST dataset...")

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        traindata = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )

        valdata = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=val_transform
        )

        # Split train into train/val
        train_size = int(0.9 * len(traindata))
        val_size = len(traindata) - train_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(traindata)))

        traindata = Subset(traindata, train_indices)
        valdata = Subset(valdata, val_indices)

        # Wrap to convert to RGB
        traindata = RGBWrapper(traindata)
        valdata = RGBWrapper(valdata)

        train_dataset = traindata
        val_dataset = valdata

    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2  # Prefetch batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, scaler, writer=None, epoch=0):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    num_triplets = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        with torch.cuda.amp.autocast():
            embeddings = model(images)

            # Mine hard triplets
            triplets = mine_hard_triplets(embeddings, labels)

            if len(triplets) == 0:
                continue

            anchors = torch.stack([embeddings[t[0]] for t in triplets])
            positives = torch.stack([embeddings[t[1]] for t in triplets])
            negatives = torch.stack([embeddings[t[2]] for t in triplets])

            loss = criterion(anchors, positives, negatives)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_triplets += len(triplets)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'triplets': len(triplets)})

    avg_loss = total_loss / len(train_loader)
    avg_triplets = num_triplets / len(train_loader)

    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Triplets/train', avg_triplets, epoch)

    return avg_loss, avg_triplets


@torch.no_grad()
def validate(model, val_loader, criterion, device, writer=None, epoch=0):
    """Validate model with mixed precision"""
    model.eval()
    total_loss = 0
    num_triplets = 0

    with torch.cuda.amp.autocast():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            embeddings = model(images)

            triplets = mine_hard_triplets(embeddings, labels)

            if len(triplets) == 0:
                continue

            anchors = torch.stack([embeddings[t[0]] for t in triplets])
            positives = torch.stack([embeddings[t[1]] for t in triplets])
            negatives = torch.stack([embeddings[t[2]] for t in triplets])

            loss = criterion(anchors, positives, negatives)

            total_loss += loss.item()
            num_triplets += len(triplets)

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    avg_triplets = num_triplets / len(val_loader) if len(val_loader) > 0 else 0

    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('Triplets/val', avg_triplets, epoch)

    return avg_loss, avg_triplets


def plot_training_curves(history, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Triplets per batch
    axes[1].plot(history['train_triplets'], label='Train Triplets', marker='o')
    axes[1].plot(history['val_triplets'], label='Val Triplets', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Avg Triplets per Batch')
    axes[1].set_title('Average Triplets per Batch')
    axes[1].legend()
    axes[1].grid(True)

    # Learning Rate
    axes[2].plot(history['learning_rate'], label='Learning Rate', marker='o', color='purple')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves to {save_path}")
    plt.close()


def train_model(args):
    """Main training function"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader = get_data_loaders(
        dataset_type=args.dataset,
        batch_size=args.batch_size
    )

    # Create model
    print(f"\nCreating model with backbone: {args.backbone}")
    pretrained = not args.from_scratch
    print(f"Using pretrained weights: {pretrained}")
    model = EmbeddingNet(
        backbone=args.backbone,
        embedding_size=args.embedding_size,
        pretrained=pretrained
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer
    criterion = TripletLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # TensorBoard writer
    run_name = f"{args.dataset}_{args.backbone}"
    if args.from_scratch:
        run_name += "_scratch"
    run_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"TensorBoard logging to: runs/{run_name}")

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_triplets': [],
        'val_triplets': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_triplets = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, writer, epoch
        )

        # Validate
        val_loss, val_triplets = validate(
            model, val_loader, criterion, device, writer, epoch
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Log learning rate to TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_triplets'].append(train_triplets)
        history['val_triplets'].append(val_triplets)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Triplets: {train_triplets:.1f} | Val Triplets: {val_triplets:.1f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"checkpoints/{args.dataset}_{args.backbone}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, model_path)
            print(f"  ✓ Saved best model to {model_path}")

        # Save checkpoint every N epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/{args.dataset}_{args.backbone}_epoch{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = f"checkpoints/{args.dataset}_{args.backbone}_final.pth"
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': history['val_loss'][-1],
        'args': vars(args),
        'history': history
    }, final_path)
    print(f"\n✓ Saved final model to {final_path}")

    # Save training history
    history_path = f"results/{args.dataset}_{args.backbone}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")

    # Plot training curves
    plot_path = f"results/{args.dataset}_{args.backbone}_training_curves.png"
    plot_training_curves(history, plot_path)

    # Close TensorBoard writer
    writer.close()
    print(f"✓ Closed TensorBoard writer")

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)

    return final_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image similarity model')

    # Data
    parser.add_argument('--dataset', type=str, default='stanford',
                        choices=['fashionmnist', 'stanford'],
                        help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')

    # Model
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v2'],
                        help='Backbone architecture')
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Size of embedding vector')

    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Margin for triplet loss')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train from scratch (no pretrained weights)')

    args = parser.parse_args()

    train_model(args)