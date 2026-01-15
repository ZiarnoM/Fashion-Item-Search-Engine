"""
Training with Backbone Fine-Tuning
This allows the model to learn product-specific features, not just generic ImageNet features
"""

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

from pytorch_metric_learning import losses, miners, distances
from stanford_products_loader import create_stanford_loaders


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


class EmbeddingNetWithFineTuning(nn.Module):
    """Embedding network with gradual backbone unfreezing"""
    
    def __init__(self, backbone='resnet50', embedding_size=128, dropout=0.5):
        super().__init__()
        
        if backbone == 'resnet50':
            base_model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            num_features = base_model.fc.in_features
            
            # Split backbone into stages for gradual unfreezing
            self.backbone_stages = nn.ModuleList([
                nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool),
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool
            ])
            
        elif backbone == 'efficientnet':
            base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_features = base_model.classifier[1].in_features
            self.backbone_stages = nn.ModuleList([nn.Sequential(*list(base_model.children())[:-1])])
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze all backbone initially
        for stage in self.backbone_stages:
            for param in stage.parameters():
                param.requires_grad = False
        
        # Embedding head (always trainable)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
    
    def forward(self, x):
        for stage in self.backbone_stages:
            x = stage(x)
        embeddings = self.embedding(x)
        return nn.functional.normalize(embeddings, p=2, dim=1)
    
    def unfreeze_last_n_stages(self, n):
        """Unfreeze the last n stages of the backbone"""
        print(f"Unfreezing last {n} backbone stage(s)")
        for stage in self.backbone_stages[-n:]:
            for param in stage.parameters():
                param.requires_grad = True
    
    def get_trainable_params(self):
        """Get count of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_data_loaders(batch_size=128, data_aug=True, dataset='stanford'):
    if dataset == 'stanford':
        return create_stanford_loaders(
            root_dir='./data/Stanford_Online_Products',
            batch_size=batch_size,
            num_workers=4
        )
    # Fashion-MNIST code omitted for brevity
    raise NotImplementedError("Only Stanford supported in this version")


def create_loss_function(loss_type='multi_similarity'):
    if loss_type == 'multi_similarity':
        return losses.MultiSimilarityLoss(alpha=2.0, beta=50, base=0.5)
    elif loss_type == 'triplet':
        return losses.TripletMarginLoss(margin=0.1, distance=distances.CosineSimilarity())
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


def train_epoch(model, loader, criterion, optimizer, device, miner=None, epoch=0, writer=None):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        embeddings = model(images)
        
        if miner:
            hard_pairs = miner(embeddings, labels)
            loss = criterion(embeddings, labels, hard_pairs)
        else:
            loss = criterion(embeddings, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    for images, labels in tqdm(loader, desc='Validation'):
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{args.dataset}_finetuned_{timestamp}')
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        data_aug=True,
        dataset=args.dataset
    )
    
    # Create model
    print(f"Creating model with fine-tuning strategy...")
    model = EmbeddingNetWithFineTuning(
        backbone=args.backbone,
        embedding_size=args.embedding_size,
        dropout=args.dropout
    ).to(device)
    
    print(f"Initial trainable parameters: {model.get_trainable_params():,}")
    
    # Loss and optimizer
    criterion = create_loss_function(args.loss_type)
    miner = miners.MultiSimilarityMiner(epsilon=0.1) if args.hard_mining else None
    
    # PHASE 1: Train only embedding head (epochs 0-5)
    # PHASE 2: Unfreeze last stage (epochs 6-10)
    # PHASE 3: Unfreeze last 2 stages (epochs 11+)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': [], 'epoch': []}
    
    print(f"\n{'='*70}")
    print("TRAINING WITH GRADUAL UNFREEZING")
    print(f"{'='*70}")
    print(f"Phase 1 (epochs 0-5):   Train embedding head only")
    print(f"Phase 2 (epochs 6-10):  Unfreeze last backbone stage")
    print(f"Phase 3 (epochs 11+):   Unfreeze last 2 stages")
    print(f"{'='*70}\n")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Gradual unfreezing strategy
        if epoch == 6:
            print("\n🔓 PHASE 2: Unfreezing last backbone stage")
            model.unfreeze_last_n_stages(1)
            # Create new optimizer with unfrozen params
            optimizer = optim.AdamW([
                {'params': model.embedding.parameters(), 'lr': args.lr},
                {'params': model.backbone_stages[-1].parameters(), 'lr': args.lr / 10}  # 10x lower LR
            ], weight_decay=args.weight_decay)
            print(f"Trainable parameters: {model.get_trainable_params():,}")
        
        elif epoch == 11:
            print("\n🔓 PHASE 3: Unfreezing last 2 backbone stages")
            model.unfreeze_last_n_stages(2)
            optimizer = optim.AdamW([
                {'params': model.embedding.parameters(), 'lr': args.lr},
                {'params': model.backbone_stages[-2].parameters(), 'lr': args.lr / 20},
                {'params': model.backbone_stages[-1].parameters(), 'lr': args.lr / 10}
            ], weight_decay=args.weight_decay)
            print(f"Trainable parameters: {model.get_trainable_params():,}")
        
        elif epoch == 0:
            # Phase 1: only embedding head
            optimizer = optim.AdamW(
                model.embedding.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, miner, epoch, writer)
        val_loss = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        history['epoch'].append(epoch)
        
        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Monitor overfitting
        gap = train_loss - val_loss
        if gap < -0.3:
            print(f"⚠️  Overfitting detected (gap={gap:.3f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = f'models/{args.dataset}_finetuned_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
                'history': history
            }, save_path)
            print(f"✓ Saved best model (val={val_loss:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping at epoch {epoch + 1}")
            break
    
    # Save history
    import json
    with open(f'results/{args.dataset}_finetuned_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    writer.close()
    print(f"\n✓ Training completed! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with backbone fine-tuning')
    
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='multi_similarity')
    parser.add_argument('--hard_mining', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--dataset', type=str, default='stanford')
    
    args = parser.parse_args()
    main(args)