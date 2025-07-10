from embedding_utilities import get_node_features
from kg_utilitics import build_knowledge_graph, adjacency_matrix  # Fixed typo
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from config import Config
from gcn_model import GCN

def train_gcn(gcn, node_feats, adj, target_embeds, class_list, epochs=200, lr=1e-3, 
              device='cpu', save_path="gcn_trained.pth", weight_decay=1e-5):
    """
    Enhanced GCN training with additional features
    """
    gcn.to(device)
    node_feats = node_feats.to(device)
    adj = adj.to(device)
    target_embeds = target_embeds.to(device)

    # Use weight decay for regularization
    optimizer = optim.Adam(gcn.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use a combination of MSE and cosine similarity loss
    mse_loss = nn.MSELoss()
    cos_loss = nn.CosineEmbeddingLoss()
    
    # Learning rate scheduler (removed verbose for compatibility)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=20)

    gcn.train()
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 50
    
    train_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = gcn(node_feats, adj)
        
        # Normalize outputs and targets for cosine similarity
        out_norm = F.normalize(out, p=2, dim=1)
        target_norm = F.normalize(target_embeds, p=2, dim=1)
        
        # Combined loss: MSE + Cosine similarity
        mse_loss_val = mse_loss(out, target_embeds)
        
        # Cosine similarity loss (we want similarity = 1, so target = 1)
        cos_target = torch.ones(out.size(0)).to(device)
        cos_loss_val = cos_loss(out_norm, target_norm, cos_target)
        
        # Total loss with weighting
        total_loss = 0.5 * mse_loss_val + 0.5 * cos_loss_val
        
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(gcn.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update learning rate and print if changed
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(total_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
        
        train_losses.append(total_loss.item())
        
        # Early stopping
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
            # Save best model
            save_model_checkpoint(gcn, node_feats, adj, class_list, save_path.replace('.pth', '_best.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss.item():.4f}, "
                  f"MSE: {mse_loss_val.item():.4f}, Cos: {cos_loss_val.item():.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Save final model with complete checkpoint
    save_model_checkpoint(gcn, node_feats, adj, class_list, save_path)
    print(f"Final GCN model saved to {save_path}")
    
    return train_losses

def save_model_checkpoint(gcn, node_feats, adj, class_list, save_path):
    """
    Save complete model checkpoint including all necessary components
    """
    checkpoint = {
        'model_state_dict': gcn.state_dict(),
        'model_config': {
            'in_dim': gcn.gcn1.linear.in_features,
            'hidden_dim': gcn.gcn1.linear.out_features,
            'out_dim': gcn.gcn2.linear.out_features,
        },
        'class_list': class_list,
        'node_features': node_feats.cpu(),
        'adjacency_matrix': adj.cpu(),
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_trained_gcn(checkpoint_path, device='cpu'):
    """
    Load trained GCN model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model
    config = checkpoint['model_config']
    gcn = GCN(config['in_dim'], config['hidden_dim'], config['out_dim'])
    gcn.load_state_dict(checkpoint['model_state_dict'])
    gcn.to(device)
    gcn.eval()
    
    # Get other components
    class_list = checkpoint['class_list']
    node_features = checkpoint['node_features'].to(device)
    adj_matrix = checkpoint['adjacency_matrix'].to(device)
    
    return gcn, class_list, node_features, adj_matrix

def evaluate_gcn(gcn, node_feats, adj, target_embeds, class_list, device='cpu'):
    """
    Evaluate the trained GCN model
    """
    gcn.eval()
    with torch.no_grad():
        out = gcn(node_feats.to(device), adj.to(device))
        target_embeds = target_embeds.to(device)
        
        # Compute metrics
        mse_loss = nn.MSELoss()(out, target_embeds)
        
        # Cosine similarity between output and target
        out_norm = F.normalize(out, p=2, dim=1)
        target_norm = F.normalize(target_embeds, p=2, dim=1)
        cos_sim = F.cosine_similarity(out_norm, target_norm, dim=1)
        avg_cos_sim = cos_sim.mean()
        
        print(f"\nEvaluation Results:")
        print(f"MSE Loss: {mse_loss.item():.4f}")
        print(f"Average Cosine Similarity: {avg_cos_sim.item():.4f}")
        print(f"Min Cosine Similarity: {cos_sim.min().item():.4f}")
        print(f"Max Cosine Similarity: {cos_sim.max().item():.4f}")
        
        # Show some examples
        print(f"\nSample similarities for first 5 classes:")
        for i in range(min(5, len(class_list))):
            print(f"{class_list[i]}: {cos_sim[i].item():.4f}")
    
    return out

def get_class_embeddings_for_inference(checkpoint_path, device='cpu'):
    """
    Load model and get normalized embeddings for zero-shot inference
    """
    gcn, class_list, node_features, adj_matrix = load_trained_gcn(checkpoint_path, device)
    
    with torch.no_grad():
        embeddings = gcn(node_features, adj_matrix)
        # Normalize for cosine similarity matching
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings, class_list

# Enhanced training script
if __name__ == "__main__":
    cfg = Config()
    
    print("Building knowledge graph...")
    G = build_knowledge_graph(cfg.class_list)
    print(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    print("Getting node features...")
    node_feats = get_node_features(cfg.class_list)   # shape: (num_classes, embedding_dim)
    print(f"Node features shape: {node_feats.shape}")
    
    print("Building adjacency matrix...")
    adj = adjacency_matrix(G, cfg.class_list)
    print(f"Adjacency matrix shape: {adj.shape}")
    
    # Use node_feats as target embeddings (autoencoder style)
    # You could also use different target embeddings here (e.g., from a different model)
    target_embeds = node_feats.clone()
    
    # Initialize GCN
    gcn = GCN(cfg.embedding_dim, cfg.gcn_hidden_dim, cfg.gcn_out_dim)
    print(f"GCN initialized: {cfg.embedding_dim} -> {cfg.gcn_hidden_dim} -> {cfg.gcn_out_dim}")
    
    # Train the model
    print("\nStarting training...")
    train_losses = train_gcn(gcn, node_feats, adj, target_embeds, cfg.class_list,
                           epochs=200, lr=1e-3, device=cfg.device, 
                           save_path="gcn_trained.pth")
    
    # Evaluate the trained model
    print("\nEvaluating trained model...")
    evaluate_gcn(gcn, node_feats, adj, target_embeds, cfg.class_list, cfg.device)
    
    # Test loading and getting embeddings for inference
    print("\nTesting inference setup...")
    class_embeddings, class_names = get_class_embeddings_for_inference("gcn_trained_best.pth", cfg.device)
    print(f"Class embeddings for inference shape: {class_embeddings.shape}")
    print(f"Classes: {class_names[:5]}...")  # Show first 5 classes
    
    print("\nTraining completed successfully!")