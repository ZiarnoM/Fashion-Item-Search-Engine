import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
from datetime import datetime

# Triplet Loss
from pytorch_metric_learning import losses, miners


# Convert grayscale to RGB
class RGBWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if img.shape[0] == 1:  # Grayscale
            img = img.repeat(3, 1, 1)
        return img, label

class EmbeddingNet(nn.Module):
    """Embedding network with pre-trained backbone"""

    def __init__(self, backbone='resnet50', embedding_size=128):
        super().__init__()

        if backbone == 'resnet50':
            base_model = torchvision.models.resnet50(pretrained=True)
            num_features = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'efficientnet':
            base_model = torchvision.models.efficientnet_b0(pretrained=True)
            num_features = base_model.classifier[1].in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Custom embedding head
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        # L2 normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


def convert_to_rgb(img):
    """Convert a grayscale image to RGB."""
    return img.convert("RGB")

def get_data_loaders(batch_size=64, data_aug=True):
    """Prepare data loaders with augmentation"""

    # Data augmentation for training
    if data_aug:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Lambda(convert_to_rgb),  # Use the top-level function
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(convert_to_rgb),  # Use the top-level function
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(convert_to_rgb),  # Use the top-level function
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Fashion-MNIST (auto-download)
    full_train = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )



    full_train = RGBWrapper(full_train)

    # Split train into train/val (80/20)
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_data, val_data = random_split(full_train, [train_size, val_size])

    # Test set
    test_data = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    test_data = RGBWrapper(test_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, miner=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)

        # Use hard triplet mining if available
        if miner:
            hard_pairs = miner(embeddings, labels)
            loss = criterion(embeddings, labels, hard_pairs)
        else:
            loss = criterion(embeddings, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0

    for images, labels in tqdm(loader, desc='Validation'):
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Tensorboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{args.backbone}_{timestamp}')

    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_aug=args.augmentation
    )

    # Model
    print(f"Creating model with {args.backbone} backbone...")
    model = EmbeddingNet(
        backbone=args.backbone,
        embedding_size=args.embedding_size
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = losses.TripletMarginLoss(margin=0.2)
    miner = miners.MultiSimilarityMiner() if args.hard_mining else None

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, miner)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = f'models/{args.backbone}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, save_path)
            print(f"✓ Saved best model to {save_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    writer.close()
    print(f"\n✓ Training completed! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet'])
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--hard_mining', action='store_true', default=True)

    args = parser.parse_args()
    main(args)