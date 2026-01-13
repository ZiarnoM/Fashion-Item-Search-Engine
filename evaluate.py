import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

from train import EmbeddingNet
from stanford_products_loader import StanfordProductsDataset


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract embeddings for all images"""
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(loader, desc='Extracting embeddings'):
        images = images.to(device)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)

    return embeddings, labels


def compute_similarity(query_emb, gallery_embs):
    """Compute cosine similarity between query and gallery"""
    # Cosine similarity (embeddings are already normalized)
    similarities = np.dot(gallery_embs, query_emb)
    return similarities


def recall_at_k(query_embs, query_labels, gallery_embs, gallery_labels, k_values=[1, 5, 10, 20]):
    """Compute Recall@K metric"""
    recalls = {k: [] for k in k_values}

    for i, (query_emb, query_label) in enumerate(zip(query_embs, query_labels)):
        # Compute similarities
        sims = compute_similarity(query_emb, gallery_embs)

        # Get top-k indices (excluding self if query is in gallery)
        top_k_indices = np.argsort(sims)[::-1]

        # Check recalls
        for k in k_values:
            top_k_labels = gallery_labels[top_k_indices[:k]]
            recalls[k].append(int(query_label in top_k_labels))

    # Average recalls
    recall_scores = {k: np.mean(recalls[k]) for k in k_values}
    return recall_scores


def mean_average_precision(query_embs, query_labels, gallery_embs, gallery_labels, k=10):
    """Compute Mean Average Precision@K"""
    aps = []

    for query_emb, query_label in zip(query_embs, query_labels):
        # Compute similarities
        sims = compute_similarity(query_emb, gallery_embs)

        # Sort by similarity
        sorted_indices = np.argsort(sims)[::-1][:k]
        sorted_labels = gallery_labels[sorted_indices]

        # Compute average precision
        relevant = (sorted_labels == query_label).astype(int)
        if relevant.sum() == 0:
            aps.append(0.0)
            continue

        precision_at_k = []
        num_relevant = 0
        for i, rel in enumerate(relevant):
            if rel:
                num_relevant += 1
                precision_at_k.append(num_relevant / (i + 1))

        ap = np.mean(precision_at_k) if precision_at_k else 0.0
        aps.append(ap)

    return np.mean(aps)


def visualize_retrieval(model, test_loader, device, num_examples=5, dataset_type='fashionmnist'):
    """Visualize retrieval results"""
    model.eval()

    # Get some test data
    test_images, test_labels = next(iter(test_loader))

    # Extract embeddings
    with torch.no_grad():
        test_images_gpu = test_images.to(device)
        embeddings = model(test_images_gpu).cpu().numpy()

    fig, axes = plt.subplots(num_examples, 6, figsize=(15, num_examples * 2.5))

    for i in range(num_examples):
        query_idx = i
        query_emb = embeddings[query_idx]
        query_label = test_labels[query_idx].item()

        # Compute similarities
        sims = compute_similarity(query_emb, embeddings)
        top_5_indices = np.argsort(sims)[::-1][1:6]  # Exclude self

        # Denormalize for visualization
        def denorm(img):
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            return torch.clamp(img, 0, 1)

        # Plot query
        ax = axes[i, 0] if num_examples > 1 else axes[0]
        query_img = denorm(test_images[query_idx]).permute(1, 2, 0).numpy()
        ax.imshow(query_img)
        ax.set_title(f'Query\nClass: {query_label}', fontsize=10)
        ax.axis('off')

        # Plot top-5 results
        for j, idx in enumerate(top_5_indices):
            ax = axes[i, j + 1] if num_examples > 1 else axes[j + 1]
            result_img = denorm(test_images[idx]).permute(1, 2, 0).numpy()
            result_label = test_labels[idx].item()
            similarity = sims[idx]

            color = 'green' if result_label == query_label else 'red'
            ax.imshow(result_img)
            ax.set_title(f'Sim: {similarity:.3f}\nClass: {result_label}',
                         fontsize=9, color=color)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('results/retrieval_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved retrieval visualization to results/retrieval_visualization.png")
    plt.close()


# RGB wrapper for Fashion-MNIST
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


def load_test_data(dataset_type='fashionmnist', batch_size=128):
    """Load test dataset"""

    if dataset_type == 'stanford':
        print("Loading Stanford Online Products test set...")

        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_dataset = StanfordProductsDataset(
            root_dir='./data/Stanford_Online_Products',
            split='test',
            transform=test_transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return test_loader

    else:  # fashionmnist
        print("Loading Fashion-MNIST test set...")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        testdata = datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        testdata = RGBWrapper(testdata)

        test_loader = DataLoader(
            testdata,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        return test_loader


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Handle both dict and Namespace args
    model_args = checkpoint['args']
    if isinstance(model_args, dict):
        backbone = model_args['backbone']
        embedding_size = model_args['embedding_size']
        dataset_type = model_args.get('dataset', 'fashionmnist')
    else:
        backbone = model_args.backbone
        embedding_size = model_args.embedding_size
        dataset_type = getattr(model_args, 'dataset', 'fashionmnist')

    # Override dataset type if specified
    if args.dataset:
        dataset_type = args.dataset

    print(f"Dataset: {dataset_type}")
    print(f"Backbone: {backbone}")
    print(f"Embedding size: {embedding_size}")

    # Load model
    model = EmbeddingNet(
        backbone=backbone,
        embedding_size=embedding_size
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    test_loader = load_test_data(dataset_type=dataset_type, batch_size=args.batch_size)

    # Extract embeddings
    print("\nExtracting embeddings from test set...")
    embeddings, labels = extract_embeddings(model, test_loader, device)

    print(f"Extracted {len(embeddings)} embeddings")
    print(f"Number of unique classes: {len(np.unique(labels))}")

    # For Stanford: use standard retrieval protocol (query != gallery)
    # For Fashion-MNIST: split test set
    if dataset_type == 'stanford':
        # Stanford has separate train/test classes
        # Use all test as both query and gallery
        query_embs = embeddings
        query_labels = labels
        gallery_embs = embeddings
        gallery_labels = labels
        print("Using all test samples as query and gallery")
    else:
        # Split test set for Fashion-MNIST
        split_idx = len(embeddings) // 2
        query_embs = embeddings[:split_idx]
        query_labels = labels[:split_idx]
        gallery_embs = embeddings[split_idx:]
        gallery_labels = labels[split_idx:]
        print(f"Query set: {len(query_embs)} samples")
        print(f"Gallery set: {len(gallery_embs)} samples")

    # Compute metrics
    print("\nComputing evaluation metrics...")

    # Recall@K
    recall_scores = recall_at_k(
        query_embs, query_labels, gallery_embs, gallery_labels,
        k_values=[1, 5, 10, 20, 50]
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS")
    print("=" * 60)
    for k, score in recall_scores.items():
        print(f"Recall@{k:>2}: {score:.4f} ({score * 100:.2f}%)")

    # MAP@K
    map_score = mean_average_precision(query_embs, query_labels, gallery_embs, gallery_labels, k=10)
    print(f"\nMean Average Precision@10: {map_score:.4f} ({map_score * 100:.2f}%)")

    # Embedding statistics
    print("\n" + "=" * 60)
    print("EMBEDDING STATISTICS")
    print("=" * 60)
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Embedding mean: {embeddings.mean():.4f}")
    print(f"Embedding std: {embeddings.std():.4f}")
    print(f"Embedding L2 norm (avg): {np.linalg.norm(embeddings, axis=1).mean():.4f}")

    # Save results
    results = {
        'model': args.model_path,
        'dataset': dataset_type,
        'backbone': backbone,
        'embedding_size': embedding_size,
        'test_samples': len(embeddings),
        'num_classes': len(np.unique(labels)),
        'recall@1': float(recall_scores[1]),
        'recall@5': float(recall_scores[5]),
        'recall@10': float(recall_scores[10]),
        'recall@20': float(recall_scores[20]),
        'recall@50': float(recall_scores[50]),
        'map@10': float(map_score),
        'total_params': sum(p.numel() for p in model.parameters()),
        'embedding_stats': {
            'mean': float(embeddings.mean()),
            'std': float(embeddings.std()),
            'l2_norm': float(np.linalg.norm(embeddings, axis=1).mean())
        }
    }

    output_file = f"results/{dataset_type}_{backbone}_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved metrics to {output_file}")

    # Visualize retrievals
    print("\nGenerating retrieval visualizations...")
    visualize_retrieval(model, test_loader, device, num_examples=5, dataset_type=dataset_type)

    print("\n" + "=" * 60)
    print("✓ EVALUATION COMPLETED!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['fashionmnist', 'stanford'],
                        help='Dataset type (if not in model checkpoint)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    args = parser.parse_args()
    main(args)