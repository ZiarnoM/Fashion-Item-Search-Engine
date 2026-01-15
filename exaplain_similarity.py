"""
Improved Grad-CAM for Metric Learning
Uses similarity-based activation to show what features make images similar
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
import cv2

from train import EmbeddingNet
from stanford_products_loader import StanfordProductsDataset


def denorm_img(img_tensor):
    """Denormalize to [0,1]"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    return torch.clamp(img, 0, 1)


class GradCAMForMetricLearning:
    """
    Grad-CAM specifically designed for metric learning models
    Shows what features contribute to similarity between images
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_img, target_embedding):
        """
        Generate CAM based on similarity to target embedding
        
        Args:
            input_img: Input image tensor (1, 3, 224, 224)
            target_embedding: Target embedding to compute similarity with
        """
        # Forward pass
        self.model.eval()
        input_img.requires_grad = True
        
        embedding = self.model(input_img)
        
        # Compute similarity score (what we want to maximize)
        similarity = F.cosine_similarity(embedding, target_embedding.unsqueeze(0), dim=1)
        
        # Backward pass
        self.model.zero_grad()
        similarity.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling on gradients (importance weights)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # ReLU to keep only positive contributions
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_difference_cam(self, query_img, similar_img, dissimilar_img):
        """
        Generate CAM showing what makes query similar to similar_img
        but different from dissimilar_img
        """
        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            query_emb = self.model(query_img)
            similar_emb = self.model(similar_img)
            dissimilar_emb = self.model(dissimilar_img)
        
        # CAM for similarity to similar image
        cam_similar = self.generate_cam(query_img, similar_emb)
        
        # CAM for similarity to dissimilar image  
        cam_dissimilar = self.generate_cam(query_img, dissimilar_emb)
        
        # Difference: what makes it similar to one but not the other
        cam_diff = np.maximum(cam_similar - cam_dissimilar, 0)
        cam_diff = (cam_diff - cam_diff.min()) / (cam_diff.max() - cam_diff.min() + 1e-8)
        
        return cam_diff


def apply_colormap_on_image(img_rgb, cam, alpha=0.5):
    """Apply heatmap on image"""
    # Resize CAM to image size
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255
    
    # Overlay
    result = alpha * heatmap + (1 - alpha) * img_rgb
    result = np.clip(result, 0, 1)
    
    return result


def extract_embeddings(model, dataset, device, max_samples=2000):
    """Extract embeddings efficiently"""
    model.eval()
    
    embeddings = []
    categories = []
    indices = []
    
    sample_indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Extracting embeddings"):
            img, _ = dataset[idx]
            img = img.unsqueeze(0).to(device)
            emb = model(img).cpu().numpy()[0]
            
            embeddings.append(emb)
            categories.append(dataset.super_labels[idx])
            indices.append(idx)
    
    return np.array(embeddings), np.array(categories), np.array(indices)


def visualize_with_improved_gradcam(model, dataset, embeddings, categories, indices,
                                    num_categories=5, device='cuda'):
    """Generate improved Grad-CAM visualizations"""
    
    # Get target layer (last conv layer of ResNet)
    if hasattr(model.backbone, '__getitem__'):
        target_layer = model.backbone[-1]  # layer4 of ResNet50
    else:
        target_layer = list(model.backbone.children())[-1]
    
    gradcam = GradCAMForMetricLearning(model, target_layer)
    
    os.makedirs('results/explain_improved', exist_ok=True)
    
    unique_cats = np.unique(categories)
    print(f"Found {len(unique_cats)} unique categories")
    
    for cat in unique_cats[:num_categories]:
        print(f"\nGenerating improved explanations for category {cat}...")
        
        cat_mask = categories == cat
        if cat_mask.sum() < 10:
            continue
        
        cat_indices = indices[cat_mask]
        cat_embeddings = embeddings[cat_mask]
        
        # Pick query
        query_local_idx = np.random.randint(len(cat_indices))
        query_idx = cat_indices[query_local_idx]
        query_emb = torch.from_numpy(cat_embeddings[query_local_idx]).float().to(device)
        
        # Get similar images
        sims = np.dot(cat_embeddings, cat_embeddings[query_local_idx])
        top_6 = np.argsort(sims)[-6:][::-1][1:]  # Top 5 excluding self
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Load query image
        query_img, _ = dataset[query_idx]
        query_img_tensor = query_img.unsqueeze(0).to(device)
        query_vis = denorm_img(query_img).permute(1, 2, 0).cpu().numpy()
        
        # Row 1: Query
        ax = plt.subplot(3, 6, 1)
        ax.imshow(query_vis)
        ax.set_title(f'Query (Cat {cat})', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Row 2: Query self-attention (what features define this image)
        ax = plt.subplot(3, 6, 7)
        try:
            cam_query = gradcam.generate_cam(query_img_tensor, query_emb)
            cam_on_img = apply_colormap_on_image(query_vis, cam_query, alpha=0.5)
            ax.imshow(cam_on_img)
            ax.set_title('Query Features', fontsize=10)
        except Exception as e:
            print(f"  Warning: {e}")
            ax.imshow(query_vis)
        ax.axis('off')
        
        # Row 3: Query raw heatmap
        ax = plt.subplot(3, 6, 13)
        try:
            im = ax.imshow(cam_query, cmap='jet')
            ax.set_title('Attention Heatmap', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
        except:
            ax.axis('off')
        ax.axis('off')
        
        # Top 5 similar images
        for i, local_idx in enumerate(top_6):
            retrieved_idx = cat_indices[local_idx]
            retrieved_emb = torch.from_numpy(cat_embeddings[local_idx]).float().to(device)
            similarity = sims[local_idx]
            
            retrieved_img, _ = dataset[retrieved_idx]
            retrieved_img_tensor = retrieved_img.unsqueeze(0).to(device)
            retrieved_vis = denorm_img(retrieved_img).permute(1, 2, 0).cpu().numpy()
            
            # Row 1: Original image
            ax = plt.subplot(3, 6, i + 2)
            ax.imshow(retrieved_vis)
            ax.set_title(f'Top {i+1} (Sim: {similarity:.3f})', fontsize=10)
            ax.axis('off')
            
            # Row 2: What makes it similar to query
            ax = plt.subplot(3, 6, i + 8)
            try:
                cam_retrieved = gradcam.generate_cam(retrieved_img_tensor, query_emb)
                cam_on_img = apply_colormap_on_image(retrieved_vis, cam_retrieved, alpha=0.5)
                ax.imshow(cam_on_img)
                ax.set_title(f'Shared Features', fontsize=10)
            except Exception as e:
                ax.imshow(retrieved_vis)
            ax.axis('off')
            
            # Row 3: Raw heatmap
            ax = plt.subplot(3, 6, i + 14)
            try:
                im = ax.imshow(cam_retrieved, cmap='jet')
                plt.colorbar(im, ax=ax, fraction=0.046)
            except:
                pass
            ax.axis('off')
        
        plt.suptitle(f'Improved Grad-CAM for Category {cat} - Similarity-based Attention',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = f'results/explain_improved/explanation_category_{cat}.png'
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
    
    model = EmbeddingNet(
        backbone=model_args.get('backbone', 'resnet50'),
        embedding_size=model_args.get('embedding_size', 256)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    print("Loading dataset...")
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = StanfordProductsDataset(args.data_dir, 'test', transform=test_transform)
    
    # Extract embeddings
    embeddings, categories, indices = extract_embeddings(
        model, dataset, device, max_samples=args.max_samples
    )
    
    # Generate visualizations
    print("\nGenerating improved Grad-CAM visualizations...")
    visualize_with_improved_gradcam(
        model, dataset, embeddings, categories, indices,
        num_categories=args.num_categories, device=device
    )
    
    print(f"\n✓ Improved explanations saved to results/explain_improved/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_dir', default='./data/Stanford_Online_Products')
    parser.add_argument('--num_categories', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=2000)
    args = parser.parse_args()
    main(args)