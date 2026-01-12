import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from collections import Counter
import json
from tqdm import tqdm
import argparse
from train import EmbeddingNet

# FashionMNIST labels
LABELS = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

plt.style.use('default')
sns.set_palette("husl")


# RGB wrapper like in your code
class RGBWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if img.shape[0] == 1:  # Grayscale
            img = img.repeat(3, 1, 1)
        return img, label

def load_data(split='train', size_limit=None):
    """Load FashionMNIST with your RGB wrapper transform."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=(split == 'train'), download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    

    
    wrapped = RGBWrapper(dataset)


    if size_limit:
        indices = np.random.choice(len(wrapped), size_limit, replace=False)
        from torch.utils.data import Subset
        wrapped = Subset(wrapped, indices)
    
    loader = torch.utils.data.DataLoader(wrapped, batch_size=256, shuffle=False, num_workers=4)
    return loader

def compute_stats(loader):
    """Basic dataset statistics."""
    all_labels = []
    for _, labels in tqdm(loader, desc="Computing stats"):
        all_labels.extend(labels.numpy())
    
    counts = Counter(all_labels)
    print("Dataset stats:")
    for i, count in sorted(counts.items()):
        print(f"  {LABELS[i]}: {count} images ({count/len(all_labels)*100:.1f}%)")
    
    return np.array(all_labels)

def extract_embeddings(model_path, loader, device, max_samples=5000):
    """Extract embeddings using your trained model (optional)."""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_args = checkpoint['args']
        model = EmbeddingNet(
            backbone=model_args['backbone'],
            embedding_size=model_args['embedding_size']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model: {model_path}")
    except:
        print("No model found; skipping embedding extraction.")
        return None, None
    
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device)
            emb = model(images).cpu().numpy()
            embeddings.append(emb)
            labels.extend(lbls.numpy())
    
    return np.vstack(embeddings)[:max_samples], np.array(labels)[:max_samples]

def plot_class_dist(labels, title="Class Distribution"):
    """Bar plot of class distribution."""
    counts = Counter(labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), ax=ax, palette='viridis')
    ax.set_xticklabels([LABELS[i] for i in counts.keys()], rotation=45)
    ax.set_title(title)
    ax.set_ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('results/images/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_embeddings_tsne(embeddings, labels, title="t-SNE Embeddings"):
    """t-SNE visualization."""
    if embeddings is None:
        print("No embeddings; skipping.")
        return
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    emb_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, ticks=range(10), label='Class', format=plt.FuncFormatter(lambda x, _: LABELS[int(x)]))
    plt.tight_layout()
    plt.savefig('results/images/tsne_embeddings.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_embeddings_umap(embeddings, labels, title="UMAP Embeddings"):
    """UMAP visualization."""
    if embeddings is None:
        print("No embeddings; skipping.")
        return
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb_2d = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, ticks=range(10), label='Class', format=plt.FuncFormatter(lambda x, _: LABELS[int(x)]))
    plt.tight_layout()
    plt.savefig('results/images/umap_embeddings.png', dpi=150, bbox_inches='tight')
    plt.show()

def sample_images(loader, n_per_class=4):
    """Grid of sample images."""
    images, labels = next(iter(loader))
    images = images[:40]  # 10 classes x 4
    labels = labels[:40]
    
    fig, axes = plt.subplots(4, 10, figsize=(20, 8))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        ax = axes[i//10, i%10]
        # Denormalize
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        img = torch.clamp(img, 0, 1)
        ax.imshow(img.permute(1,2,0))
        ax.set_title(LABELS[lbl], fontsize=10)
        ax.axis('off')
    plt.suptitle('Sample Images (Train Set)', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/images/sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()

def main(model_path=None, data_split='test', max_emb_samples=5000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    loader = load_data(split=data_split, size_limit=None)
    print(f"Loaded {data_split} set: {len(loader.dataset)} images")
    
    # Basic stats
    labels = compute_stats(loader)
    plot_class_dist(labels, f"{data_split.capitalize()} Class Distribution")
    
    # Sample images
    sample_loader = load_data(split=data_split, size_limit=40)
    sample_images(sample_loader)
    
    # Embeddings (if model provided)
    embeddings = None
    if model_path:
        embeddings, emb_labels = extract_embeddings(model_path, loader, device, max_emb_samples)
        plot_class_dist(emb_labels, "Embedding Sample Class Distribution")
        plot_embeddings_tsne(embeddings, emb_labels)
        plot_embeddings_umap(embeddings, emb_labels)
    
    print("Saved plots: class_distribution.png, sample_images.png, [tsne/umap_embeddings.png]")
    print("Open in Jupyter or run `jupyter notebook visualize_fashionmnist.py` for interactive version.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/resnet50_best.pth", help="Path to trained model")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--max-emb", type=int, default=5000, help="Max embedding samples")
    args = parser.parse_args()
    main(args.model, args.split, args.max_emb)
