import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
from config import Config
from kg_utilitics import build_knowledge_graph, adjacency_matrix
from embedding_utilities import get_node_features
from gcn_model import GCN
from inference import load_detector, prepare_image, extract_global_feature
from train_gcn import get_class_embeddings_for_inference
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchvision import transforms
from train_gcn import get_class_embeddings_for_inference  # Import the function to load trained GCN

# Step 1: Load trained GCN and get class embeddings
print("\n=== Loading Trained GCN Model ===")
try:
    class_embeddings, class_list = get_class_embeddings_for_inference("gcn_trained.pth", device)
    print(f"✓ Loaded GCN embeddings: {class_embeddings.shape}")
    print(f"✓ Classes: {class_list}")
except Exception as e:
    print(f"Error loading GCN model: {e}")
    print("Make sure you've trained the GCN model first!")

    

def classify_image(image_path, device):
    cfg = Config()

    # Load and preprocess image
    orig_image = Image.open(image_path).convert("RGB")
    #transform = T.Compose([
    #    T.ToTensor(),
     #   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #])

    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(orig_image).unsqueeze(0).to(device)

    # Load detector and GCN embeddings
    detector = load_detector(device)
    # Prepare image and extract global feature
    image_list = prepare_image(image_tensor)
    global_feat = extract_global_feature(detector, image_list)  # (1, feat_dim)

    # Compute cosine similarity to each class embedding
    similarities = F.cosine_similarity(global_feat, class_embeddings, dim=1)  # (num_classes,)

    # Sort and display top-k
    topk = torch.topk(similarities, k=6)
    print("\nTop predictions:")
    for idx, score in zip(topk.indices, topk.values):
        #print(f"{cfg.class_list[idx]}: {score.item():.4f}")
        print(f"{cfg.class_list[idx]}")


if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = sys.argv[1] if len(sys.argv) > 1 else "dishwash.webp"
    classify_image(image_path, device)
