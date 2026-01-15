"""
Explainability for Image Retrieval Model using Grad-CAM++
Shows which parts of images the model focuses on for similarity
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

# For Grad-CAM
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from train import EmbeddingNet
from stanford_products_loader import StanfordProductsDataset


def denorm_img(img_tensor):
    """Denormalize image tensor to [0,1] for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    return torch.clamp(img, 0, 1)


def compute_similarity(query_emb, gallery_embs):
    """Cosine similarity"""
    return np.dot(gallery_embs, query_emb)


def extract_embeddings(model, dataset, device, max_samples=2000):
    """Extract embeddings from dataset"""
    model.eval()

    embeddings = []
    product_ids = []
    categories = []
    indices = []

    # Sample indices
    sample_indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Extracting embeddings"):
            img, label = dataset[idx]
            img = img.unsqueeze(0).to(device)
            emb = model(img).cpu().numpy()[0]

            embeddings.append(emb)
            product_ids.append(label)
            categories.append(dataset.super_labels[idx])
            indices.append(idx)

    return (
        np.array(embeddings),
        np.array(product_ids),
        np.array(categories),
        np.array(indices)
    )


class EmbeddingModelWrapper(torch.nn.Module):
    """Wrapper to make the model compatible with Grad-CAM"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embedding = None

    def forward(self, x):
        # Forward through model
        emb = self.model(x)
        self.embedding = emb

        # For Grad-CAM, we need a scalar output
        # Use L2 norm as a proxy (all embeddings should have similar norm)
        return emb.norm(dim=1, keepdim=True)


def generate_gradcam_for_pair(model, img1_tensor, img2_tensor, target_layers, device):
    """
    Generate Grad-CAM for both images in a similar pair

    Args:
        img1_tensor: Query image (1, 3, 224, 224)
        img2_tensor: Retrieved image (1, 3, 224, 224)
    """
    # Prepare images for visualization (RGB in [0,1])
    img1_vis = denorm_img(img1_tensor[0]).permute(1, 2, 0).cpu().numpy()
    img2_vis = denorm_img(img2_tensor[0]).permute(1, 2, 0).cpu().numpy()

    # Create Grad-CAM for image 1
    cam1 = GradCAMPlusPlus(model, target_layers=target_layers)

    # Reshape target for Grad-CAM (it expects [batch, classes])
    def custom_forward(x):
        return model(x)  # Returns (batch, 1) from wrapper

    # Generate CAM for query
    img1_input = img1_tensor.to(device)
    img1_input.requires_grad = True

    grayscale_cam1 = cam1(input_tensor=img1_input, targets=None)
    cam1_on_image = show_cam_on_image(img1_vis, grayscale_cam1[0], use_rgb=True)

    # Generate CAM for retrieved image
    img2_input = img2_tensor.to(device)
    img2_input.requires_grad = True

    grayscale_cam2 = cam1(input_tensor=img2_input, targets=None)
    cam2_on_image = show_cam_on_image(img2_vis, grayscale_cam2[0], use_rgb=True)

    return img1_vis, img2_vis, cam1_on_image, cam2_on_image


def visualize_explanations(model, dataset, embeddings, product_ids, categories, indices,
                           num_categories=5, device='cuda'):
    """Generate explanation visualizations for multiple categories"""

    # Wrap model for Grad-CAM
    model_wrapper = EmbeddingModelWrapper(model)
    target_layers = [model.backbone[-1]]  # Last layer of ResNet

    os.makedirs('results/explain', exist_ok=True)

    # Get unique categories
    unique_cats = np.unique(categories)
    print(f"Found {len(unique_cats)} unique categories")

    selected_cats = unique_cats[:min(num_categories, len(unique_cats))]

    for cat in selected_cats:
        print(f"\nGenerating explanations for category {cat}...")

        # Get samples from this category
        cat_mask = categories == cat
        if cat_mask.sum() < 10:
            print(f"  Skipping category {cat} (too few samples)")
            continue

        cat_indices = indices[cat_mask]
        cat_embeddings = embeddings[cat_mask]
        cat_products = product_ids[cat_mask]

        # Pick a random query
        query_local_idx = np.random.randint(len(cat_indices))
        query_idx = cat_indices[query_local_idx]
        query_emb = cat_embeddings[query_local_idx]
        query_product = cat_products[query_local_idx]

        # Find similar images (exclude same product)
        valid_mask = cat_products != query_product
        valid_embeddings = cat_embeddings[valid_mask]
        valid_indices = cat_indices[valid_mask]

        if len(valid_embeddings) < 5:
            print(f"  Not enough different products in category {cat}")
            continue

        # Compute similarities
        sims = compute_similarity(query_emb, valid_embeddings)
        top_5 = np.argsort(sims)[-5:][::-1]  # Top 5 in descending order

        # Create visualization
        fig = plt.figure(figsize=(20, 8))

        # Load query image
        query_img, _ = dataset[query_idx]
        query_img_tensor = query_img.unsqueeze(0)

        # Row 1: Original images
        # Row 2: Grad-CAM heatmaps

        # Query image
        ax = plt.subplot(2, 6, 1)
        query_vis = denorm_img(query_img).permute(1, 2, 0).cpu().numpy()
        ax.imshow(query_vis)
        ax.set_title(f'Query\n(Cat {cat})', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Query Grad-CAM
        ax = plt.subplot(2, 6, 7)
        try:
            cam = GradCAMPlusPlus(model_wrapper, target_layers=target_layers)
            query_input = query_img_tensor.to(device)
            query_input.requires_grad = True
            grayscale_cam = cam(input_tensor=query_input, targets=None)
            cam_image = show_cam_on_image(query_vis, grayscale_cam[0], use_rgb=True)
            ax.imshow(cam_image)
        except Exception as e:
            print(f"  Warning: Grad-CAM failed for query: {e}")
            ax.imshow(query_vis)
        ax.set_title('Query Attention', fontsize=10)
        ax.axis('off')

        # Top 5 retrieved images
        for i, local_idx in enumerate(top_5):
            retrieved_idx = valid_indices[local_idx]
            similarity = sims[local_idx]

            # Load retrieved image
            retrieved_img, _ = dataset[retrieved_idx]
            retrieved_img_tensor = retrieved_img.unsqueeze(0)

            # Original image
            ax = plt.subplot(2, 6, i + 2)
            retrieved_vis = denorm_img(retrieved_img).permute(1, 2, 0).cpu().numpy()
            ax.imshow(retrieved_vis)
            ax.set_title(f'Top {i + 1}\nSim: {similarity:.3f}', fontsize=10)
            ax.axis('off')

            # Grad-CAM
            ax = plt.subplot(2, 6, i + 8)
            try:
                cam = GradCAMPlusPlus(model_wrapper, target_layers=target_layers)
                retrieved_input = retrieved_img_tensor.to(device)
                retrieved_input.requires_grad = True
                grayscale_cam = cam(input_tensor=retrieved_input, targets=None)
                cam_image = show_cam_on_image(retrieved_vis, grayscale_cam[0], use_rgb=True)
                ax.imshow(cam_image)
            except Exception as e:
                print(f"  Warning: Grad-CAM failed for result {i + 1}: {e}")
                ax.imshow(retrieved_vis)
            ax.set_title(f'Attention', fontsize=10)
            ax.axis('off')

        plt.suptitle(f'Visual Explanations for Category {cat} - What the model focuses on',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = f'results/explain/explanation_category_{cat}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved to {output_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint.get('args', {})

    backbone = model_args.get('backbone', 'resnet50')
    embedding_size = model_args.get('embedding_size', 256)

    print(f"  Backbone: {backbone}")
    print(f"  Embedding size: {embedding_size}")

    model = EmbeddingNet(backbone=backbone, embedding_size=embedding_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load dataset
    print("\nLoading test dataset...")
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = StanfordProductsDataset(
        root_dir=args.data_dir,
        split='test',
        transform=test_transform
    )
    print(f"  Test samples: {len(test_dataset)}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings, product_ids, categories, indices = extract_embeddings(
        model, test_dataset, device, max_samples=args.max_samples
    )
    print(f"  Extracted {len(embeddings)} embeddings")

    # Generate visualizations
    print("\nGenerating Grad-CAM explanations...")
    visualize_explanations(
        model, test_dataset, embeddings, product_ids, categories, indices,
        num_categories=args.num_categories, device=device
    )

    print(f"\n{'=' * 70}")
    print("✓ Explanations generated successfully!")
    print(f"  Output directory: results/explain/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate visual explanations for retrieval model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--data_dir', type=str, default='./data/Stanford_Online_Products',
                        help='Path to dataset')
    parser.add_argument('--num_categories', type=int, default=5,
                        help='Number of categories to explain')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Max samples to extract embeddings from')

    args = parser.parse_args()
    main(args)