import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from collections import Counter
from tqdm import tqdm
from PIL import Image
from train import EmbeddingNet

plt.style.use('default')
sns.set_palette("husl")

# Define categories for Stanford Online Products dataset
CATEGORIES = [
    'bicycle', 'cabinet', 'chair', 'coffee maker', 'fan', 'kettle',
    'lamp', 'mug', 'sofa', 'stapler', 'table', 'toaster'
]

def load_stanford_dataset(data_dir, split='train', transform=None):
    """Load Stanford Online Products dataset."""
    txt_file = os.path.join(data_dir, f'Ebay_{split}.txt')
    print(f"Loading {split} data from {txt_file}")
    with open(txt_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    images, labels = [], []
    for line in lines:
        _, class_id, super_class_id, path = line.strip().split()
        class_id, super_class_id = int(class_id), int(super_class_id)
        img_path = os.path.join(data_dir, path)
        images.append(img_path)
        labels.append(super_class_id - 1)  # Convert to 0-based index

    return images, labels

def preprocess_image(img_path, transform):
    """Load and preprocess a single image."""
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = torch.clamp(img, 0, 1)  # Clip values to [0, 1] range
    return img

def compute_stats(labels):
    """Compute and display dataset statistics."""
    counts = Counter(labels)
    print("Dataset stats:")
    for i, count in sorted(counts.items()):
        print(f"  {CATEGORIES[i]}: {count} images ({count / len(labels) * 100:.1f}%)")
    return np.array(labels)

def plot_class_dist(labels, title="Class Distribution"):
    """Bar plot of class distribution."""
    counts = Counter(labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), ax=ax, palette='viridis', hue=None,legend=False)
    ax.set_xticks(range(len(counts.keys())))
    ax.set_xticklabels([CATEGORIES[i] for i in counts.keys()], rotation=45)
    ax.set_title(title)
    ax.set_ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('results/images/stanford_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def sample_images(images, labels, transform, n_per_class=4):
    """Display a grid of sample images."""
    sampled = {i: [] for i in range(len(CATEGORIES))}
    for img_path, label in zip(images, labels):
        if len(sampled[label]) < n_per_class:
            sampled[label].append(img_path)
        if all(len(v) == n_per_class for v in sampled.values()):
            break

    fig, axes = plt.subplots(len(CATEGORIES), n_per_class, figsize=(n_per_class * 3, len(CATEGORIES) * 3))
    for i, (label, img_paths) in enumerate(sampled.items()):
        for j, img_path in enumerate(img_paths):
            img = preprocess_image(img_path, transform)
            img = img.permute(1, 2, 0).numpy()
            axes[i, j].imshow(img)
            axes[i, j].set_title(CATEGORIES[label])
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig('results/images/stanford_sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()

def extract_embeddings(model_path, images, transform, device, max_samples=5000):
    """Extract embeddings using a trained model."""
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

    embeddings, labels = [], []
    with torch.no_grad():
        for img_path, label in tqdm(zip(images, labels), total=len(images), desc="Extracting embeddings"):
            img = preprocess_image(img_path, transform).unsqueeze(0).to(device)
            emb = model(img).cpu().numpy()
            embeddings.append(emb)
            labels.append(label)
            if len(embeddings) >= max_samples:
                break

    return np.vstack(embeddings), np.array(labels)

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
    plt.colorbar(scatter, ax=ax, ticks=range(len(CATEGORIES)), label='Class', format=plt.FuncFormatter(lambda x, _: CATEGORIES[int(x)]))
    plt.tight_layout()
    plt.savefig('results/images/stanford_tsne_embeddings.png', dpi=150, bbox_inches='tight')
    plt.show()

def main(data_dir, model_path=None, split='train', max_emb_samples=5000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images, labels = load_stanford_dataset(data_dir, split, transform)
    print(f"Loaded {split} set: {len(images)} images")

    # Basic stats
    compute_stats(labels)
    plot_class_dist(labels, f"{split.capitalize()} Class Distribution")

    # Sample images
    sample_images(images, labels, transform)

    # Embeddings (if model provided)
    if model_path:
        embeddings, emb_labels = extract_embeddings(model_path, images, transform, device, max_emb_samples)
        if embeddings is not None:
            plot_embeddings_tsne(embeddings, emb_labels)
        else:
            print("Skipping embedding visualization due to missing embeddings.")

    print("Saved plots: class_distribution.png, sample_images.png, tsne_embeddings.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to Stanford dataset directory")
    parser.add_argument("--model", default=None, help="Path to trained model")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--max-emb", type=int, default=5000, help="Max embedding samples")
    args = parser.parse_args()
    main(args.data_dir, args.model, args.split, args.max_emb)