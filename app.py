import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gradio as gr
from torchvision import datasets
import random

from train import EmbeddingNet

class SearchEngine:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device,weights_only=False)
        model_args = checkpoint['args']
        
        self.model = EmbeddingNet(
            backbone=model_args.backbone,
            embedding_size=model_args.embedding_size
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load gallery database
        print("Building search database...")
        self._build_database()
        
    def _build_database(self):
        """Build embedding database from dataset"""
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True)
        
        # Store all images and labels
        self.gallery_images = []
        self.gallery_labels = []
        
        for img, label in test_data:
            # Convert to RGB
            img_rgb = img.convert('RGB')
            self.gallery_images.append(img_rgb)
            self.gallery_labels.append(label)
        
        # Extract embeddings for all gallery images
        print(f"Extracting embeddings for {len(self.gallery_images)} images...")
        self.gallery_embeddings = []
        
        with torch.no_grad():
            for img in self.gallery_images:
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                # Convert grayscale to RGB
                if img_tensor.shape[1] == 1:
                    img_tensor = img_tensor.repeat(1, 3, 1, 1)
                embedding = self.model(img_tensor)
                self.gallery_embeddings.append(embedding.cpu().numpy())
        
        self.gallery_embeddings = np.vstack(self.gallery_embeddings)
        print(f"✓ Database ready with {len(self.gallery_images)} images")
    
    def search(self, query_image, top_k=5):
        """Search for similar images"""
        # Convert to RGB if needed
        if query_image.mode != 'RGB':
            query_image = query_image.convert('RGB')
        
        # Extract query embedding
        img_tensor = self.transform(query_image).unsqueeze(0).to(self.device)
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            query_embedding = self.model(img_tensor).cpu().numpy()
        
        # Compute similarities (cosine similarity)
        similarities = np.dot(self.gallery_embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_k_indices:
            results.append({
                'image': self.gallery_images[idx],
                'label': self.gallery_labels[idx],
                'similarity': float(similarities[idx])
            })
        
        return results

# Fashion-MNIST class names
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Initialize search engine
print("Initializing search engine...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
search_engine = SearchEngine('models/resnet50_best.pth', device=device)

def search_interface(query_image, top_k):
    """Gradio interface function"""
    if query_image is None:
        return None, "Please upload an image"
    
    # Convert numpy array to PIL Image if needed
    if isinstance(query_image, np.ndarray):
        query_image = Image.fromarray(query_image)
    
    # Search
    results = search_engine.search(query_image, top_k=top_k)
    
    # Prepare output images and descriptions
    output_images = []
    descriptions = []
    
    for i, result in enumerate(results):
        output_images.append(result['image'])
        label_name = CLASS_NAMES[result['label']]
        similarity = result['similarity']
        descriptions.append(f"**Rank {i+1}**: {label_name} (Similarity: {similarity:.3f})")
    
    # Create gallery output
    description_text = "\n\n".join(descriptions)
    
    return output_images, description_text

def random_example():
    """Get random example from dataset"""
    idx = random.randint(0, len(search_engine.gallery_images) - 1)
    img = search_engine.gallery_images[idx]
    label = CLASS_NAMES[search_engine.gallery_labels[idx]]
    return img, f"Random example: {label}"

# Create Gradio interface
with gr.Blocks(title="Fashion Item Search Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Fashion Item Search Engine
        
        Upload an image of a fashion item (clothing, shoes, bags) and find similar items from the database.
        
        **How it works:**
        - Uses deep learning to extract visual features from images
        - Finds similar items based on visual similarity
        - Powered by ResNet50 + Triplet Loss metric learning
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Image(type="pil", label="Upload Query Image")
            top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, 
                                     label="Number of Results")
            search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
            random_btn = gr.Button("🎲 Try Random Example", variant="secondary")
            
            gr.Markdown("### Tips:")
            gr.Markdown("- Upload clear images of fashion items")
            gr.Markdown("- Works best with clothing, shoes, and accessories")
            gr.Markdown("- Adjust 'Number of Results' to see more matches")
        
        with gr.Column(scale=2):
            result_gallery = gr.Gallery(label="Search Results", columns=5, rows=1, height="auto")
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
    
    # Examples
    gr.Examples(
        examples=[
            ["data/FashionMNIST/raw/t10k-images-idx3-ubyte", 5],
        ],
        inputs=[query_input, top_k_slider],
        label="Example Queries (click to load random samples)"
    )

# Launch
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Fashion Search Engine...")
    print("="*60 + "\n")
    demo.launch(share=True, server_name="0.0.0.0")