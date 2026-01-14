# explain.py - Standalone XAI for your Stanford ResNet50 MultiSimilarity model
import argparse
import json
import gc
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from train import EmbeddingNet  # Reuse your model class
from stanford_products_loader import StanfordProductsDataset
import os

# Copy these helpers from evaluate.py
def load_image_by_idx(dataset, idx):
    img, _ = dataset[idx]
    return img

def denorm_img(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    return torch.clamp(img, 0, 1)

def computesimilarity(query_emb, gallery_embs):
    similarities = np.dot(gallery_embs, query_emb)
    return similarities

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model (same as evaluate.py)
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint.get('args', {})
    backbone = model_args.get('backbone', 'resnet50')
    embedding_size = model_args.get('embedding_size', 256)
    model = EmbeddingNet(backbone=backbone, embedding_size=embedding_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # print("Model loaded:")
    # print(model)

    # Setup GradCAMpp on backbone
    cam_extractor = GradCAMpp(model.backbone, target_layer='7.0.conv3')

    # Load small test batch for viz (like --vizonly)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = StanfordProductsDataset('data/Stanford_Online_Products', 'test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Extract embeddings (reuse your logic)
    embeddings, product_ids, categories = extract_embeddings_with_metadata(model, test_loader, device)
    torch.cuda.empty_cache(); gc.collect()

    # Truncate for speed
    N = 1000
    embeddings = embeddings[:N]
    product_ids = product_ids[:N]
    categories = categories[:N]

    #ensure results dir
    os.makedirs('results/explain', exist_ok=True)

    # Viz per category (like mode2)
    unique_cats = np.unique(categories)
    print("Unique categories found:", unique_cats)

    for cat in tqdm(unique_cats[:args.num_categories], desc="Generating explanations"):
        cat_mask = categories == cat
        cat_embs = embeddings[cat_mask]
        if len(cat_embs) < 10: continue

        # Pick random query
        query_idx = np.random.choice(np.where(cat_mask)[0])
        query_emb = torch.from_numpy(cat_embs[0]).float().to(device)  # First as example

        # Compute top-5 gallery (exclude same product)
        gallery_embs = embeddings[~np.isin(product_ids, product_ids[query_idx])]  # Simple exclude
        sims = computesimilarity(cat_embs[0], gallery_embs)
        top_indices = np.argsort(sims)[-6:]  # Query + top5

        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        query_img = denorm_img(load_image_by_idx(test_dataset, query_idx)).permute(1,2,0).cpu().numpy()

        # Query CAM (self-similarity proxy)
        img_q = load_image_by_idx(test_dataset, query_idx).unsqueeze(0).to(device)
        with torch.enable_grad():
            out = model(img_q)
            proxy_score = F.cosine_similarity(out, query_emb.unsqueeze(0)).sum()
            proxy_score.backward(retain_graph=True)
        cams = cam_extractor(img_q, scores=None)
        heatmap_q = cams[0].squeeze(0).cpu().sigmoid()
        axes[0,0].imshow(query_img)
        axes[0,0].imshow(heatmap_q, cmap='jet', alpha=0.5)
        axes[0,0].set_title('Query (Self-Sim CAM)')
        axes[0,0].axis('off')

        # Top-5 retrievals + CAMs
        for j, g_idx in enumerate(top_indices[1:]):  # Skip self
            g_img = denorm_img(load_image_by_idx(test_dataset, g_idx)).permute(1,2,0).cpu().numpy()
            img_g = load_image_by_idx(test_dataset, g_idx).unsqueeze(0).to(device)
            with torch.enable_grad():
                out_g = model(img_g)
                proxy_score = F.cosine_similarity(out_g, query_emb.unsqueeze(0)).sum()
                proxy_score.backward(retain_graph=True)
            cams_g = cam_extractor(img_g, scores=None)
            heatmap_g = cams_g[0].squeeze(0).cpu().sigmoid()
            sim_val = sims[g_idx]
            axes[0, j+1].imshow(g_img)
            axes[0, j+1].imshow(heatmap_g, cmap='jet', alpha=0.5)
            axes[0, j+1].set_title(f'Top{j+1} Sim:{sim_val:.3f}')
            axes[0, j+1].axis('off')

            # Pairwise comparison subplot (optional)
            axes[1, j].imshow(query_img)
            axes[1, j].axis('off')

        plt.suptitle(f'Grad-CAM++ Explanations: Category {cat}')
        plt.tight_layout()
        output_path = f'results/explain/explanations_category_{cat}.png'
        tqdm.write(f"Saving explanation to: {output_path}")
        # print(f"Saving explanation to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    print("Explanations saved to results/explanations_category_*.png")

# Paste your extract_embeddings_with_metadata from evaluate.py here (or import if modularized)
def extract_embeddings_with_metadata(model, loader, device):
    model.eval()
    all_embs, all_pids, all_cats = [], [], []
    for images, labels in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        embs = model(images).detach().cpu().numpy()
        all_embs.append(embs)
        # Add product_ids, categories logic from your evaluate.py
        # Placeholder: assume loader.dataset has them
        batch_pids = labels.numpy()  # Adapt to your dataset
        batch_cats = labels.numpy()  # Adapt
        all_pids.extend(batch_pids)
        all_cats.extend(batch_cats)
    return np.vstack(all_embs), np.array(all_pids), np.array(all_cats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to best.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_categories', type=int, default=5, help='Categories to explain')
    args = parser.parse_args()
    main(args)
