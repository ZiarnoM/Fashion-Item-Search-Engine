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


def get_data_loaders(batch_size=64, data_aug=True):
    """Prepare data loaders with augmentation"""

    # Data augmentation for training
    if data_aug:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Fashion-MNIST (auto-download)
    full_train = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        # transform=train_transform
        transform=transforms.ToTensor()
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


def train_epoch(model, loader, criterion, optimizer, device, miner=None, epoch=0, writer=None):
    """Train for one epoch with detailed logging"""
    model.train()
    total_loss = 0
    batch_losses = []

    # Track gradient norms
    grad_norms = []

    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
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

        # Calculate gradient norm (for monitoring)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        optimizer.step()

        batch_losses.append(loss.item())
        total_loss += loss.item()

        # Log to tensorboard every 50 batches
        if writer and batch_idx % 50 == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('Batch/loss', loss.item(), global_step)
            writer.add_scalar('Batch/grad_norm', total_norm, global_step)

            # Log embedding statistics
            emb_mean = embeddings.mean().item()
            emb_std = embeddings.std().item()
            writer.add_scalar('Embeddings/mean', emb_mean, global_step)
            writer.add_scalar('Embeddings/std', emb_std, global_step)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'grad_norm': f'{total_norm:.2f}'
        })

    avg_grad_norm = sum(grad_norms) / len(grad_norms)

    return total_loss / len(loader), batch_losses, avg_grad_norm


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=0, writer=None):
    """Validate model with detailed metrics"""
    model.eval()
    total_loss = 0

    # Track embedding statistics
    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(loader, desc='Validation'):
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        total_loss += loss.item()

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    # Compute embedding statistics
    all_embeddings = torch.cat(all_embeddings, dim=0)
    emb_mean = all_embeddings.mean().item()
    emb_std = all_embeddings.std().item()
    emb_l2_norm = torch.norm(all_embeddings, p=2, dim=1).mean().item()

    # Log to tensorboard
    if writer:
        writer.add_scalar('Validation/embedding_mean', emb_mean, epoch)
        writer.add_scalar('Validation/embedding_std', emb_std, epoch)
        writer.add_scalar('Validation/embedding_l2_norm', emb_l2_norm, epoch)

        # Log embedding distribution
        writer.add_histogram('Validation/embeddings', all_embeddings, epoch)

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

    # Store training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'grad_norm': [],
        'epoch': []
    }

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, batch_losses, grad_norm = train_epoch(
            model, train_loader, criterion, optimizer, device, miner,
            epoch=epoch, writer=writer
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch=epoch, writer=writer)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        history['grad_norm'].append(grad_norm)
        history['epoch'].append(epoch)

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        writer.add_scalar('Training/grad_norm', grad_norm, epoch)

        # Log batch losses distribution
        import numpy as np
        writer.add_histogram('Training/batch_losses', np.array(batch_losses), epoch)

        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}, Grad Norm: {grad_norm:.2f}")

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
                'args': args,
                'history': history
            }, save_path)
            print(f"✓ Saved best model to {save_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Save training history as JSON
    import json
    history_path = f'results/{args.backbone}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")

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