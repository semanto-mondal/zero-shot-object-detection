from sentence_transformers import SentenceTransformer
import torch


def get_node_features(classes, model_name="KennethEnevoldsen/dfm-sentence-encoder-small"):
    embedder = SentenceTransformer(model_name)
    features = embedder.encode(classes, normalize_embeddings=True)
    return torch.tensor(features, dtype=torch.float)
