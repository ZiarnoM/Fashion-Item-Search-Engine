import torch
import gc
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

torch.cuda.empty_cache()
gc.collect()


@torch.no_grad()
def extract_embeddings(model, loader, device, return_product_ids=False):
    """Extract embeddings for all images

    Args:
        return_product_ids: If True, also return the original product IDs from dataset
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_product_ids = []

    for images, labels in tqdm(loader, desc='Extracting embeddings'):
        images = images.to(device)
        embeddings = model(images)
        all_embeddings.append(embeddings.detach().cpu().numpy())
        all_labels.append(labels.numpy())

        # Get product IDs if requested (from dataset's original labels)
        if return_product_ids and hasattr(loader.dataset, 'original_product_ids'):
            batch_product_ids = [loader.dataset.original_product_ids[loader.dataset.labels[idx]]
                                 for idx in range(len(labels))]
            all_product_ids.append(np.array(batch_product_ids))

    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)

    if return_product_ids and all_product_ids:
        product_ids = np.concatenate(all_product_ids)
        return embeddings, labels, product_ids

    return embeddings, labels


def compute_similarity(query_emb, gallery_embs):
    """Compute cosine similarity between query and gallery"""
    # Cosine similarity (embeddings are already normalized)
    similarities = np.dot(gallery_embs, query_emb)
    return similarities


def recall_at_k(query_embs, query_labels, gallery_embs, gallery_labels, k_values=[1, 5, 10, 20],
                exclude_same=False, query_ids=None, gallery_ids=None):
    """Compute Recall@K metric

    Args:
        exclude_same: If True, exclude exact same product matches (for search engine evaluation)
        query_ids: Product IDs for query set (to exclude same product)
        gallery_ids: Product IDs for gallery set (to exclude same product)
    """
    recalls = {k: [] for k in k_values}

    for i, (query_emb, query_label) in enumerate(zip(query_embs, query_labels)):
        # Compute similarities
        sims = compute_similarity(query_emb, gallery_embs)

        # Get sorted indices
        sorted_indices = np.argsort(sims)[::-1]

        # If excluding same product, filter out matches with same product ID
        if exclude_same and query_ids is not None and gallery_ids is not None:
            query_product_id = query_ids[i]
            # Keep only results that are different products
            valid_mask = gallery_ids[sorted_indices] != query_product_id
            sorted_indices = sorted_indices[valid_mask]

        # Check recalls
        for k in k_values:
            top_k_labels = gallery_labels[sorted_indices[:k]]
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


def mean_average_precision_exclude_same(query_embs, query_labels, query_ids,
                                        gallery_embs, gallery_labels, gallery_ids, k=10):
    """Compute Mean Average Precision@K excluding same product matches"""
    aps = []

    for query_emb, query_label, query_id in zip(query_embs, query_labels, query_ids):
        # Compute similarities
        sims = compute_similarity(query_emb, gallery_embs)

        # Sort by similarity
        sorted_indices = np.argsort(sims)[::-1]

        # Exclude same product
        valid_mask = gallery_ids[sorted_indices] != query_id
        sorted_indices = sorted_indices[valid_mask][:k]
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


def visualize_retrieval(model, test_loader, device, num_examples=5, dataset_type="fashionmnist",
                        query_embs=None, query_labels=None, gallery_embs=None, gallery_labels=None,
                        query_images=None):
    """
    Visualize retrieval results.
    If precomputed embeddings/labels provided, use them (for mode 2).
    Otherwise, compute on-the-fly from test_loader (for product-level).
    """
    model.eval()

    # If no precomputed data, compute embeddings from test batch
    if query_embs is None or gallery_embs is None:
        test_images, test_labels = next(iter(test_loader))
        with torch.no_grad():
            test_images_gpu = test_images.to(device)
            embeddings = model(test_images_gpu).cpu().numpy()
        query_embs = embeddings
        gallery_embs = embeddings
        query_labels = test_labels
        gallery_labels = test_labels
        query_images = test_images
    else:
        if query_images is None:
            # Fallback: get batch if no images provided
            test_images, _ = next(iter(test_loader))
        else:
            test_images = query_images
        test_labels = query_labels

    # For mode 2 category-level: pick diverse categories
    all_labels = query_labels if isinstance(query_labels, np.ndarray) else query_labels.numpy()
    unique_labels = np.unique(all_labels)
    chosen_labels = np.random.choice(unique_labels,
                                     size=min(num_examples, len(unique_labels)),
                                     replace=False)

    fig, axes = plt.subplots(len(chosen_labels), 6, figsize=(15, len(chosen_labels) * 2.5))
    if len(chosen_labels) == 1:
        axes = np.expand_dims(axes, 0)

    for i, cat in enumerate(chosen_labels):
        # Pick one random query from this category
        candidate_indices = np.where(all_labels == cat)[0]
        query_idx = np.random.choice(candidate_indices)
        query_emb = query_embs[query_idx]
        query_label = int(query_labels[query_idx].item())

        # Compute similarities
        sims = compute_similarity(query_emb, gallery_embs)

        # Exclude same product if product IDs available (mode 2)
        exclude_same = False
        if 'queryproductids' in locals() and 'galleryproductids' in locals():
            q_id = queryproductids[query_idx]
            valid_mask = galleryproductids != q_id
            exclude_same = True
        else:
            valid_mask = np.ones_like(sims, dtype=bool)

        sorted_indices = np.argsort(sims[valid_mask])[::-1]
        gallery_indices = np.where(valid_mask)[0][sorted_indices[:5]]  # top 5

        # Denormalize function
        def denorm_img(img):
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            return torch.clamp(img, 0, 1)

        # Plot query
        ax = axes[i, 0]
        query_img = denorm_img(test_images[query_idx]).permute(1, 2, 0).numpy()
        ax.imshow(query_img)
        ax.set_title(f"Query cat {query_label}", fontsize=10)
        ax.axis("off")

        # Plot top 5 results
        for j, idx in enumerate(gallery_indices):
            ax = axes[i, j + 1]
            result_label = int(gallery_labels[idx].item())
            result_img = denorm_img(test_images[idx]).permute(1, 2, 0).numpy()
            color = "green" if result_label == query_label else "red"
            sim_val = sims[idx]
            ax.imshow(result_img)
            ax.set_title(f"{sim_val:.3f}\ncat {result_label}", fontsize=9, color=color)
            ax.axis("off")

    plt.tight_layout()

    # Save with mode indicator
    mode_str = "_mode2_categories" if exclude_same else "_product_level"
    save_path = f"results/retrieval_visualization{mode_str}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved retrieval visualization to {save_path}")
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


def load_test_data(dataset_type='fashionmnist', batch_size=128, return_categories=False):
    """Load test dataset

    Args:
        return_categories: If True, return a dataset that tracks both products and categories
    """

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
            root_dir='./data',
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


def extract_embeddings_with_metadata(model, loader, device):
    """Extract embeddings along with product IDs and categories"""
    model.eval()
    all_embeddings = []
    all_product_ids = []  # Mapped product IDs (0 to N)
    all_categories = []  # Categories

    dataset = loader.dataset
    has_categories = hasattr(dataset, 'super_labels')

    # Build index mapping for the current dataset split
    if hasattr(dataset, '_all_image_paths'):
        # This is a training dataset split (train or val)
        product_ids_list = []
        categories_list = []
        for img_path in dataset.image_paths:
            idx = dataset._all_image_paths.index(img_path)
            product_ids_list.append(dataset._all_labels[idx])
            if has_categories:
                categories_list.append(dataset._all_super_labels[idx] if hasattr(dataset, '_all_super_labels')
                                       else dataset.super_labels[idx])
    else:
        # Regular dataset
        product_ids_list = dataset.labels
        categories_list = dataset.super_labels if has_categories else dataset.labels

    idx = 0
    for images, labels in tqdm(loader, desc='Extracting embeddings with metadata'):
        images = images.to(device)
        embeddings = model(images)
        all_embeddings.append(embeddings.detach().cpu().numpy())

        batch_size = len(labels)
        all_product_ids.extend(product_ids_list[idx:idx + batch_size])
        if has_categories:
            all_categories.extend(categories_list[idx:idx + batch_size])
        else:
            all_categories.extend(labels.numpy())
        idx += batch_size

    embeddings = np.vstack(all_embeddings)
    product_ids = np.array(all_product_ids)
    categories = np.array(all_categories)

    torch.cuda.empty_cache()  # Add at end

    return embeddings, product_ids, categories


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

    # Extract embeddings with metadata
    print("\nExtracting embeddings from test set...")
    if args.viz_only:
        print("Viz-only mode: computing embeddings for small batch only...")
        test_batch_loader = DataLoader(test_loader.dataset, batch_size=128, shuffle=True, num_workers=4)
        embeddings, productids, categories = extract_embeddings_with_metadata(model, test_batch_loader, device)
        torch.cuda.empty_cache()
        gc.collect()
        # Truncate to first 1000 for speed
        truncate = 1000
        embeddings = embeddings[:truncate]
        productids = productids[:truncate]
        categories = categories[:truncate]
        print(f"Truncated to {truncate} samples for viz-only")
    else:
        embeddings, productids, categories = extract_embeddings_with_metadata(model, test_loader, device)

    print(f"Extracted {len(embeddings)} embeddings")
    print(f"Number of unique products: {len(np.unique(product_ids))}")
    print(f"Number of unique categories: {len(np.unique(categories))}")

    # For Stanford: use standard retrieval protocol
    # For Fashion-MNIST: split test set
    if dataset_type == 'stanford':
        query_embs = embeddings
        query_product_ids = product_ids
        query_categories = categories
        gallery_embs = embeddings
        gallery_product_ids = product_ids
        gallery_categories = categories
        print("Using all test samples as query and gallery")
    else:
        # Split test set for Fashion-MNIST
        split_idx = len(embeddings) // 2
        query_embs = embeddings[:split_idx]
        query_product_ids = product_ids[:split_idx]
        query_categories = categories[:split_idx]
        gallery_embs = embeddings[split_idx:]
        gallery_product_ids = product_ids[split_idx:]
        gallery_categories = categories[split_idx:]
        print(f"Query set: {len(query_embs)} samples")
        print(f"Gallery set: {len(gallery_embs)} samples")

    # # Compute metrics
    # print("\n" + "=" * 60)
    # print("EVALUATION MODE 1: PRODUCT-LEVEL RETRIEVAL")
    # print("(Finding same product - metric learning evaluation)")
    # print("=" * 60)
    #
    # # Product-level Recall@K (same as before)
    # product_recall_scores = recall_at_k(
    #     query_embs, query_product_ids, gallery_embs, gallery_product_ids,
    #     k_values=[1, 5, 10, 20, 50]
    # )
    #
    # for k, score in product_recall_scores.items():
    #     print(f"Product Recall@{k:>2}: {score:.4f} ({score * 100:.2f}%)")
    #
    # # Product-level MAP@K
    # product_map_score = mean_average_precision(
    #     query_embs, query_product_ids, gallery_embs, gallery_product_ids, k=10
    # )
    # print(f"Product MAP@10: {product_map_score:.4f} ({product_map_score * 100:.2f}%)")

    # NEW: Category-level retrieval (for search engine)
    print("\n" + "=" * 60)
    print("EVALUATION MODE 2: CATEGORY-LEVEL RETRIEVAL (Search Engine)")
    print("(Finding similar products, excluding same product)")
    print("=" * 60)

    # Category Recall@K - exclude same product matches
    category_recall_scores = recall_at_k(
        query_embs, query_categories, gallery_embs, gallery_categories,
        k_values=[1, 5, 10, 20, 50],
        exclude_same=True,
        query_ids=query_product_ids,
        gallery_ids=gallery_product_ids
    )

    for k, score in category_recall_scores.items():
        print(f"Category Recall@{k:>2}: {score:.4f} ({score * 100:.2f}%)")

    # Category MAP@K - exclude same product
    category_map_score = mean_average_precision_exclude_same(
        query_embs, query_categories, query_product_ids,
        gallery_embs, gallery_categories, gallery_product_ids, k=10
    )
    print(f"Category MAP@10: {category_map_score:.4f} ({category_map_score * 100:.2f}%)")

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
        'num_unique_products': len(np.unique(product_ids)),
        'num_categories': len(np.unique(categories)),

        # # Product-level metrics (metric learning evaluation)
        # 'product_metrics': {
        #     'recall@1': float(product_recall_scores[1]),
        #     'recall@5': float(product_recall_scores[5]),
        #     'recall@10': float(product_recall_scores[10]),
        #     'recall@20': float(product_recall_scores[20]),
        #     'recall@50': float(product_recall_scores[50]),
        #     'map@10': float(product_map_score),
        # },

        # Category-level metrics (search engine evaluation - excludes same product)
        'category_metrics': {
            'recall@1': float(category_recall_scores[1]),
            'recall@5': float(category_recall_scores[5]),
            'recall@10': float(category_recall_scores[10]),
            'recall@20': float(category_recall_scores[20]),
            'recall@50': float(category_recall_scores[50]),
            'map@10': float(category_map_score),
        },

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
    # print("Printing retrieval visualizations (product-level)...")
    # visualize_retrieval(model, test_loader, device, num_examples=5, dataset_type=dataset_type)

    # Get a test batch for images (same as product-level uses)
    test_images, _ = next(iter(test_loader))

    print("Printing retrieval visualizations (mode 2 - category-level, diverse categories)...")
    visualize_retrieval(
        model, test_loader, device, num_examples=10, dataset_type=dataset_type,
        query_embs=query_embs, query_labels=query_categories,
        gallery_embs=gallery_embs, gallery_labels=gallery_categories,
        query_images=test_images  # Pass images for visualization
    )

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
    parser.add_argument("--viz_only", action="store_true", help="Skip metrics, just do visualization")
    args = parser.parse_args()
    main(args)
