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
    """Extract embeddings for all images with mixed precision for speed"""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_product_ids = []

    # Use automatic mixed precision for faster inference
    with torch.cuda.amp.autocast():
        for images, labels in tqdm(loader, desc='Extracting embeddings'):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.detach().cpu().numpy())
            all_labels.append(labels.numpy())

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
    similarities = np.dot(gallery_embs, query_emb)
    return similarities


def recall_at_k_fast(query_embs, query_labels, gallery_embs, gallery_labels, k_values=[1, 5, 10, 20],
                     exclude_same=False, query_ids=None, gallery_ids=None, chunk_size=1000):
    """
    OPTIMIZED: Vectorized Recall@K computation
    
    Args:
        chunk_size: Process queries in chunks to manage memory
    """
    recalls = {k: [] for k in k_values}
    max_k = max(k_values)
    
    num_queries = len(query_embs)
    num_gallery = len(gallery_embs)
    
    print(f"Computing recall for {num_queries} queries against {num_gallery} gallery items...")
    
    # Process in chunks to avoid memory issues
    for chunk_start in tqdm(range(0, num_queries, chunk_size), desc="Computing Recall@K"):
        chunk_end = min(chunk_start + chunk_size, num_queries)
        
        # Get chunk of queries
        query_chunk = query_embs[chunk_start:chunk_end]
        label_chunk = query_labels[chunk_start:chunk_end]
        
        # VECTORIZED: Compute all similarities at once (chunk_size x num_gallery)
        similarities = query_chunk @ gallery_embs.T  # Matrix multiplication
        
        # Handle exclude_same if needed
        if exclude_same and query_ids is not None and gallery_ids is not None:
            query_id_chunk = query_ids[chunk_start:chunk_end]
            # Create mask: (chunk_size, num_gallery)
            same_product_mask = query_id_chunk[:, np.newaxis] == gallery_ids[np.newaxis, :]
            # Set similarity of same products to very negative value so they don't appear in top-k
            similarities[same_product_mask] = -np.inf
        
        # Get top-k indices for all queries in chunk at once
        # argsort returns ascending, we want descending (highest similarity first)
        # Using argpartition for faster partial sort when k << num_gallery
        if max_k < num_gallery // 2:
            # Faster for small k: partition instead of full sort
            top_k_indices = np.argpartition(-similarities, kth=max_k-1, axis=1)[:, :max_k]
            # Sort just the top-k
            top_k_sims = np.take_along_axis(similarities, top_k_indices, axis=1)
            sorted_within_topk = np.argsort(-top_k_sims, axis=1)
            top_k_indices = np.take_along_axis(top_k_indices, sorted_within_topk, axis=1)
        else:
            # Full sort if k is large
            top_k_indices = np.argsort(-similarities, axis=1)[:, :max_k]
        
        # Get labels for top-k results
        top_k_labels = gallery_labels[top_k_indices]  # (chunk_size, max_k)
        
        # Check if query label is in top-k for each k value
        query_labels_expanded = label_chunk[:, np.newaxis]  # (chunk_size, 1)
        
        for k in k_values:
            # Check if any of top-k matches query label
            matches = (top_k_labels[:, :k] == query_labels_expanded).any(axis=1)
            recalls[k].extend(matches.tolist())
    
    # Average recalls
    recall_scores = {k: np.mean(recalls[k]) for k in k_values}
    return recall_scores


def precision_at_k_fast(query_embs, query_labels, gallery_embs, gallery_labels, k=5, chunk_size=1000):
    """OPTIMIZED: Vectorized Precision@K computation"""
    all_precisions = []
    
    num_queries = len(query_embs)
    
    for chunk_start in tqdm(range(0, num_queries, chunk_size), desc=f"Computing Precision@{k}"):
        chunk_end = min(chunk_start + chunk_size, num_queries)
        
        query_chunk = query_embs[chunk_start:chunk_end]
        label_chunk = query_labels[chunk_start:chunk_end]
        
        # Compute similarities
        similarities = query_chunk @ gallery_embs.T
        
        # Get top-k
        top_k_indices = np.argpartition(-similarities, kth=k-1, axis=1)[:, :k]
        top_k_labels = gallery_labels[top_k_indices]
        
        # Compute precision for each query
        precisions = (top_k_labels == label_chunk[:, np.newaxis]).sum(axis=1) / k
        all_precisions.extend(precisions.tolist())
    
    return np.mean(all_precisions)


def mean_average_precision_fast(query_embs, query_labels, gallery_embs, gallery_labels, k=10, chunk_size=1000):
    """OPTIMIZED: Vectorized Mean Average Precision@K"""
    all_aps = []
    
    num_queries = len(query_embs)
    
    for chunk_start in tqdm(range(0, num_queries, chunk_size), desc="Computing MAP@K"):
        chunk_end = min(chunk_start + chunk_size, num_queries)
        
        query_chunk = query_embs[chunk_start:chunk_end]
        label_chunk = query_labels[chunk_start:chunk_end]
        
        # Compute similarities
        similarities = query_chunk @ gallery_embs.T
        
        # Get top-k indices
        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        top_k_labels = gallery_labels[top_k_indices]
        
        # Compute AP for each query
        relevant = (top_k_labels == label_chunk[:, np.newaxis]).astype(float)
        
        # For each query, compute average precision
        for i in range(len(query_chunk)):
            rel = relevant[i]
            if rel.sum() == 0:
                all_aps.append(0.0)
                continue
            
            # Positions where relevant items appear (1-indexed)
            relevant_positions = np.where(rel)[0] + 1
            # Precision at each relevant position
            precisions_at_relevant = np.arange(1, len(relevant_positions) + 1) / relevant_positions
            ap = precisions_at_relevant.mean()
            all_aps.append(ap)
    
    return np.mean(all_aps)


def mean_average_precision_exclude_same_fast(query_embs, query_labels, query_ids,
                                              gallery_embs, gallery_labels, gallery_ids, 
                                              k=10, chunk_size=1000):
    """OPTIMIZED: Vectorized MAP@K excluding same product matches"""
    all_aps = []
    
    num_queries = len(query_embs)
    
    for chunk_start in tqdm(range(0, num_queries, chunk_size), desc="Computing MAP@K (exclude same)"):
        chunk_end = min(chunk_start + chunk_size, num_queries)
        
        query_chunk = query_embs[chunk_start:chunk_end]
        label_chunk = query_labels[chunk_start:chunk_end]
        query_id_chunk = query_ids[chunk_start:chunk_end]
        
        # Compute similarities
        similarities = query_chunk @ gallery_embs.T
        
        # Mask out same products
        same_product_mask = query_id_chunk[:, np.newaxis] == gallery_ids[np.newaxis, :]
        similarities[same_product_mask] = -np.inf
        
        # Get top-k indices
        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        top_k_labels = gallery_labels[top_k_indices]
        
        # Compute AP for each query
        relevant = (top_k_labels == label_chunk[:, np.newaxis]).astype(float)
        
        for i in range(len(query_chunk)):
            rel = relevant[i]
            if rel.sum() == 0:
                all_aps.append(0.0)
                continue
            
            relevant_positions = np.where(rel)[0] + 1
            precisions_at_relevant = np.arange(1, len(relevant_positions) + 1) / relevant_positions
            ap = precisions_at_relevant.mean()
            all_aps.append(ap)
    
    return np.mean(all_aps)


def visualize_retrieval(model, test_loader, device, num_examples=5, dataset_type="fashionmnist",
                        query_embs=None, query_labels=None, gallery_embs=None, gallery_labels=None,
                        query_product_ids=None, gallery_product_ids=None):
    """Visualize retrieval results"""
    model.eval()

    if query_embs is None or gallery_embs is None:
        test_images, test_labels = next(iter(test_loader))
        with torch.no_grad():
            test_images_gpu = test_images.to(device)
            embeddings = model(test_images_gpu).cpu().numpy()
        query_embs = embeddings
        gallery_embs = embeddings
        query_labels = test_labels.numpy()
        gallery_labels = test_labels.numpy()
        use_precomputed = False
    else:
        use_precomputed = True

    all_labels = query_labels if isinstance(query_labels, np.ndarray) else query_labels.numpy()
    unique_labels = np.unique(all_labels)
    chosen_labels = np.random.choice(unique_labels,
                                     size=min(num_examples, len(unique_labels)),
                                     replace=False)

    fig, axes = plt.subplots(len(chosen_labels), 6, figsize=(15, len(chosen_labels) * 2.5))
    if len(chosen_labels) == 1:
        axes = np.expand_dims(axes, 0)

    def load_image_by_idx(idx):
        dataset = test_loader.dataset
        img, _ = dataset[idx]
        return img

    def denorm_img(img):
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        return torch.clamp(img, 0, 1)

    for i, cat in enumerate(chosen_labels):
        candidate_indices = np.where(all_labels == cat)[0]
        query_idx = np.random.choice(candidate_indices)
        query_emb = query_embs[query_idx]
        query_label = int(query_labels[query_idx])

        # Compute similarities (vectorized for all gallery)
        sims = gallery_embs @ query_emb

        # Exclude same product if product IDs available
        exclude_same = False
        if query_product_ids is not None and gallery_product_ids is not None:
            q_id = query_product_ids[query_idx]
            valid_mask = gallery_product_ids != q_id
            exclude_same = True
        else:
            valid_mask = np.ones_like(sims, dtype=bool)

        sorted_indices = np.argsort(sims[valid_mask])[::-1]
        gallery_indices = np.where(valid_mask)[0][sorted_indices[:5]]

        # Plot query
        ax = axes[i, 0]
        if use_precomputed:
            query_img = load_image_by_idx(query_idx)
        else:
            query_img = test_images[query_idx]
        query_img = denorm_img(query_img).permute(1, 2, 0).numpy()
        ax.imshow(query_img)
        ax.set_title(f"Query\nCategory: {query_label}", fontsize=10)
        ax.axis("off")

        # Plot top 5 results
        for j, idx in enumerate(gallery_indices):
            ax = axes[i, j + 1]
            result_label = int(gallery_labels[idx])

            if use_precomputed:
                result_img = load_image_by_idx(idx)
            else:
                result_img = test_images[idx]
            result_img = denorm_img(result_img).permute(1, 2, 0).numpy()

            color = "green" if result_label == query_label else "red"
            sim_val = sims[idx]
            ax.imshow(result_img)
            ax.set_title(f"Sim: {sim_val:.3f}\nCat: {result_label}", fontsize=9, color=color)
            ax.axis("off")

    plt.tight_layout()

    mode_str = "_mode2_categories" if exclude_same else "_product_level"
    model_name = args.model_path.split('/')[-1].replace('.pth', '')
    save_path = f"results/retrieval_visualization_{model_name}_{mode_str}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved retrieval visualization to {save_path}")
    plt.close()


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


def extract_embeddings_with_metadata(model, loader, device):
    """Extract embeddings along with product IDs and categories - OPTIMIZED"""
    model.eval()
    all_embeddings = []
    all_product_ids = []
    all_categories = []

    dataset = loader.dataset
    has_categories = hasattr(dataset, 'super_labels')

    # Build index mapping
    if hasattr(dataset, '_all_image_paths'):
        product_ids_list = []
        categories_list = []
        for img_path in dataset.image_paths:
            idx = dataset._all_image_paths.index(img_path)
            product_ids_list.append(dataset._all_labels[idx])
            if has_categories:
                categories_list.append(dataset._all_super_labels[idx] if hasattr(dataset, '_all_super_labels')
                                       else dataset.super_labels[idx])
    else:
        product_ids_list = dataset.labels
        categories_list = dataset.super_labels if has_categories else dataset.labels

    idx = 0
    
    # Use mixed precision for faster inference
    with torch.cuda.amp.autocast():
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

    torch.cuda.empty_cache()

    return embeddings, product_ids, categories


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    model_args = checkpoint['args']
    if isinstance(model_args, dict):
        backbone = model_args['backbone']
        embedding_size = model_args['embedding_size']
        dataset_type = model_args.get('dataset', 'fashionmnist')
    else:
        backbone = model_args.backbone
        embedding_size = model_args.embedding_size
        dataset_type = getattr(model_args, 'dataset', 'fashionmnist')

    if args.dataset:
        dataset_type = args.dataset

    print(f"Dataset: {dataset_type}")
    print(f"Backbone: {backbone}")
    print(f"Embedding size: {embedding_size}")

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
        test_batch_loader = DataLoader(test_loader.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        embeddings, product_ids, categories = extract_embeddings_with_metadata(model, test_batch_loader, device)
        torch.cuda.empty_cache()
        gc.collect()
        truncate = 1000
        embeddings = embeddings[:truncate]
        product_ids = product_ids[:truncate]
        categories = categories[:truncate]
        print(f"Truncated to {truncate} samples for viz-only")
    else:
        embeddings, product_ids, categories = extract_embeddings_with_metadata(model, test_loader, device)

    print(f"Extracted {len(embeddings)} embeddings")
    print(f"Number of unique products: {len(np.unique(product_ids))}")
    print(f"Number of unique categories: {len(np.unique(categories))}")

    # Compute per-category distances
    for cat in range(1, 13):
        cat_mask = categories == cat
        cat_embeddings = embeddings[cat_mask]
        if len(cat_embeddings) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(cat_embeddings, metric='cosine')
            print(f"Category {cat}: Mean distance={distances.mean():.4f}, Std={distances.std():.4f}")

    # Split data
    if dataset_type == 'stanford':
        query_embs = embeddings
        query_product_ids = product_ids
        query_categories = categories
        gallery_embs = embeddings
        gallery_product_ids = product_ids
        gallery_categories = categories
        print("Using all test samples as query and gallery")
    else:
        split_idx = len(embeddings) // 2
        query_embs = embeddings[:split_idx]
        query_product_ids = product_ids[:split_idx]
        query_categories = categories[:split_idx]
        gallery_embs = embeddings[split_idx:]
        gallery_product_ids = product_ids[split_idx:]
        gallery_categories = categories[split_idx:]
        print(f"Query set: {len(query_embs)} samples")
        print(f"Gallery set: {len(gallery_embs)} samples")

    # OPTIMIZED EVALUATION
    print("\n" + "=" * 60)
    print("EVALUATION MODE 2: CATEGORY-LEVEL RETRIEVAL (Search Engine)")
    print("(Finding similar products, excluding same product)")
    print("=" * 60)

    # Use optimized functions with chunking
    category_recall_scores = recall_at_k_fast(
        query_embs, query_categories, gallery_embs, gallery_categories,
        k_values=[1, 5, 10, 20],
        exclude_same=True,
        query_ids=query_product_ids,
        gallery_ids=gallery_product_ids,
        chunk_size=args.chunk_size
    )

    for k, score in category_recall_scores.items():
        print(f"Category Recall@{k:>2}: {score:.4f} ({score * 100:.2f}%)")

    category_map_score = mean_average_precision_exclude_same_fast(
        query_embs, query_categories, query_product_ids,
        gallery_embs, gallery_categories, gallery_product_ids, 
        k=10,
        chunk_size=args.chunk_size
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

    print("\n" + "=" * 60)
    print("EVALUATION: PRECISION@K")
    print("=" * 60)

    k_values = [1, 5, 10, 20]
    for k in k_values:
        precision = precision_at_k_fast(query_embs, query_categories, gallery_embs, gallery_categories, 
                                        k=k, chunk_size=args.chunk_size)
        print(f"Precision@{k}: {precision:.4f} ({precision * 100:.2f}%)")

    # Save results
    results = {
        'model': args.model_path,
        'dataset': dataset_type,
        'backbone': backbone,
        'embedding_size': embedding_size,
        'test_samples': len(embeddings),
        'num_unique_products': len(np.unique(product_ids)),
        'num_categories': len(np.unique(categories)),
        'category_metrics': {
            'recall@1': float(category_recall_scores[1]),
            'recall@5': float(category_recall_scores[5]),
            'recall@10': float(category_recall_scores[10]),
            'recall@20': float(category_recall_scores[20]),
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
    visualize_retrieval(
        model, test_loader, device, num_examples=10, dataset_type=dataset_type,
        query_embs=query_embs, query_labels=query_categories,
        gallery_embs=gallery_embs, gallery_labels=gallery_categories,
        query_product_ids=query_product_ids,
        gallery_product_ids=gallery_product_ids
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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Chunk size for similarity computation (reduce if OOM)')
    parser.add_argument("--viz_only", action="store_true", 
                        help="Skip metrics, just do visualization")
    args = parser.parse_args()
    main(args)
