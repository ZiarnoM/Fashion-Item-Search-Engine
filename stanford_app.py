import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gradio as gr
import random
import os
from pathlib import Path

from train import EmbeddingNet
from stanford_products_loader import StanfordProductsDataset


class StanfordSearchEngine:
    def __init__(self, model_path, data_root='./data/Stanford_Online_Products', device='cpu'):
        self.device = device
        self.data_root = data_root

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_args = checkpoint['args']

        self.model = EmbeddingNet(
            backbone=model_args['backbone'],
            embedding_size=model_args['embedding_size']
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded: {model_args['backbone']} with {model_args['embedding_size']}-dim embeddings")

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load gallery database
        print("Building search database...")
        self._build_database()

    def _build_database(self):
        """Build embedding database from Stanford test set"""
        print("Loading Stanford Online Products test set...")

        # Load test dataset (without transform for storing original images)
        test_dataset = StanfordProductsDataset(
            root_dir=self.data_root,
            split='test',
            transform=None
        )

        print(f"Test set size: {len(test_dataset)} images")

        # Sample a subset
        max_gallery_size = 5000
        indices = list(range(len(test_dataset)))

        if len(test_dataset) > max_gallery_size:
            print(f"Sampling {max_gallery_size} images for gallery (out of {len(test_dataset)})")
            indices = random.sample(indices, max_gallery_size)

        # Store images and labels
        self.gallery_images = []
        self.gallery_labels = []
        self.gallery_embeddings = []

        # Extract embeddings in batches
        print(f"Extracting embeddings for {len(indices)} gallery images...")
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_images = []
                batch_tensors = []

                for idx in batch_indices:
                    # Get original image (no transform)
                    img, label = test_dataset[idx]
                    self.gallery_images.append(img)
                    self.gallery_labels.append(label)

                    # Apply transform for embedding extraction
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)

                # Extract embeddings
                batch = torch.stack(batch_tensors).to(self.device)
                embeddings = self.model(batch)
                self.gallery_embeddings.append(embeddings.cpu().numpy())

        self.gallery_embeddings = np.vstack(self.gallery_embeddings)
        print(f"✓ Database ready with {len(self.gallery_images)} images")
        print(f"  Embedding shape: {self.gallery_embeddings.shape}")

    def search(self, query_image, top_k=5):
        """Search for similar products"""
        # Convert to RGB if needed
        if query_image.mode != 'RGB':
            query_image = query_image.convert('RGB')

        # Extract query embedding
        img_tensor = self.transform(query_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            query_embedding = self.model(img_tensor).cpu().numpy()

        # Compute similarities (cosine similarity)
        similarities = np.dot(self.gallery_embeddings, query_embedding.T).flatten()

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        # Prepare results
        results = []
        for rank, idx in enumerate(top_k_indices):
            results.append({
                'image': self.gallery_images[idx],
                'similarity': float(similarities[idx]),
                'rank': rank + 1
            })

        return results

    def get_random_image(self):
        """Get a random image from gallery"""
        idx = random.randint(0, len(self.gallery_images) - 1)
        return self.gallery_images[idx], idx


# Initialize search engine
print("=" * 60)
print("Initializing Stanford Online Products Search Engine")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_PATH = 'models/stanford_efficientnet_b0_best.pth'
DATA_ROOT = './data/Stanford_Online_Products'

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f" Model not found at {MODEL_PATH}")
    print("Please update MODEL_PATH in the script to point to your trained model")
    print("Available checkpoints:")
    if os.path.exists('checkpoints'):
        for f in os.listdir('checkpoints'):
            if f.endswith('.pth'):
                print(f"  - checkpoints/{f}")
    search_engine = None
else:
    search_engine = StanfordSearchEngine(MODEL_PATH, DATA_ROOT, device=device)


def search_interface(query_image, top_k):
    """Gradio interface function"""
    if search_engine is None:
        return None, " Search engine not initialized. Please check model path."

    if query_image is None:
        return None, "Please upload an image"

    # Convert numpy array to PIL Image if needed
    if isinstance(query_image, np.ndarray):
        query_image = Image.fromarray(query_image)

    results = search_engine.search(query_image, top_k=top_k)

    output_images = []
    descriptions = []

    for result in results:
        output_images.append(result['image'])
        descriptions.append(f"**Rank {result['rank']}**: Similarity {result['similarity']:.3f}")

    description_text = "\n\n".join(descriptions)

    return output_images, description_text


def random_example():
    """Get random example from gallery"""
    if search_engine is None:
        return None, " Search engine not initialized"

    img, idx = search_engine.get_random_image()
    return img, "Random example from gallery"


# Create Gradio interface
with gr.Blocks(title="Stanford Products Search Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Stanford Online Products Search Engine

        Upload an image of a product and find visually similar items from the Stanford Online Products dataset.

        **How it works:**
        - Uses EfficientNet-B0 backbone trained with triplet loss
        - Extracts 128-dimensional embeddings for semantic similarity
        - Finds similar products using cosine similarity in embedding space

        **Dataset:** Stanford Online Products (120k images, 22k products, 12 categories)
        """
    )

    if search_engine is None:
        gr.Markdown(
            """
            ##  Setup Required

            Please update the following paths in `app_stanford.py`:
            - `MODEL_PATH`: Path to your trained model (e.g., 'checkpoints/stanford_efficientnet_b0_best.pth')
            - `DATA_ROOT`: Path to Stanford Online Products dataset
            """
        )

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Image(type="pil", label="Upload Query Image")
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Results"
            )
            search_btn = gr.Button(" Search", variant="primary", size="lg")
            random_btn = gr.Button(" Try Random Example", variant="secondary")

            gr.Markdown("### Tips:")
            gr.Markdown("- Upload clear product images")
            gr.Markdown("- Works best with frontal views")
            gr.Markdown("- Covers categories like furniture, bikes, lamps, etc.")
            gr.Markdown("- Adjust 'Number of Results' to see more matches")

        with gr.Column(scale=2):
            result_gallery = gr.Gallery(
                label="Search Results",
                columns=5,
                rows=2,
                height="auto",
                object_fit="contain"
            )
            result_text = gr.Markdown()

    # Connect interface
    search_btn.click(
        fn=search_interface,
        inputs=[query_input, top_k_slider],
        outputs=[result_gallery, result_text]
    )

    random_btn.click(
        fn=random_example,
        outputs=[query_input, result_text]
    )

    gr.Markdown(
        """
        ---
        ### Model Information
        - **Architecture:** EfficientNet-B0 + Custom Embedding Head
        - **Training:** Triplet Loss with Semi-Hard Negative Mining
        - **Performance:** ~60% Recall@1, ~65% mAP@10
        - **Parameters:** 4.7M (lightweight and efficient)
        """
    )

# Launch
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Starting Stanford Products Search Engine...")
    print("=" * 60 + "\n")

    if search_engine is not None:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860
        )
    else:
        print(" Cannot launch: Search engine not initialized")
        print("Please check MODEL_PATH and DATA_ROOT in the script")