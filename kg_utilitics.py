import networkx as nx
import torch
import requests
import time
from typing import List, Dict, Tuple

def get_conceptnet_neighbors(concept: str, class_list: List[str], lang: str = 'en', max_retries: int = 3) -> nx.Graph:
    """
    Enhanced version with error handling and retries
    """
    concept_uri = f'/c/{lang}/{concept.lower().replace(" ", "_")}'
    url = f'https://api.conceptnet.io{concept_uri}?offset=0&limit=1000'
    G = nx.Graph()
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {concept}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
            else:
                print(f"Failed to fetch ConceptNet data for {concept} after {max_retries} attempts")
                return G

    # Process edges
    class_list_lower = [cls.lower().replace(" ", "_") for cls in class_list]
    class_mapping = {cls.lower().replace(" ", "_"): cls for cls in class_list}
    
    for edge in data.get('edges', []):
        try:
            rel = edge['rel']['label']
            start = edge['start']['label'].lower().replace(" ", "_")
            end = edge['end']['label'].lower().replace(" ", "_")
            
            # Map back to original class names
            if start in class_list_lower and end in class_list_lower and start != end:
                start_orig = class_mapping[start]
                end_orig = class_mapping[end]
                G.add_edge(start_orig, end_orig, relation=rel)
                
        except (KeyError, AttributeError) as e:
            continue  # Skip malformed edges
    
    return G

def build_knowledge_graph(class_list: List[str]) -> nx.Graph:
    """
    Enhanced knowledge graph building with progress tracking
    """
    KG = nx.Graph()
    print(f"Building knowledge graph for {len(class_list)} classes...")
    
    for i, concept in enumerate(class_list):
        print(f"Processing {i+1}/{len(class_list)}: {concept}")
        subgraph = get_conceptnet_neighbors(concept, class_list)
        KG.add_edges_from(subgraph.edges(data=True))
        KG.add_nodes_from(subgraph.nodes())
        
        # Small delay to be nice to the API
        time.sleep(0.1)
    
    # Ensure all classes are in the graph as nodes (even if isolated)
    for cls in class_list:
        if cls not in KG.nodes():
            KG.add_node(cls)
    
    return KG

def create_semantic_fallback_graph(class_list: List[str]) -> nx.Graph:
    """
    Create fallback connections based on semantic similarity
    """
    from embedding_utilities import get_node_features
    import torch.nn.functional as F
    
    # Get embeddings for all classes
    embeddings = get_node_features(class_list)
    
    # Compute pairwise similarities
    similarities = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(class_list)
    
    # Add edges for highly similar classes (top 20% similarities)
    threshold = torch.quantile(similarities[similarities != 1.0], 0.8)  # Top 20%
    
    for i in range(len(class_list)):
        for j in range(i+1, len(class_list)):
            if similarities[i, j] > threshold:
                G.add_edge(class_list[i], class_list[j], 
                          relation='semantic_similarity', 
                          weight=similarities[i, j].item())
    
    return G

def enhanced_build_knowledge_graph(class_list: List[str], 
                                 min_edges_ratio: float = 0.1,
                                 use_semantic_fallback: bool = True) -> nx.Graph:
    """
    Build knowledge graph with fallback strategies
    """
    print("Building knowledge graph with ConceptNet...")
    KG = build_knowledge_graph(class_list)
    
    min_expected_edges = max(1, int(len(class_list) * min_edges_ratio))
    
    print(f"ConceptNet result: {KG.number_of_nodes()} nodes, {KG.number_of_edges()} edges")
    print(f"Minimum expected edges: {min_expected_edges}")
    
    if KG.number_of_edges() < min_expected_edges and use_semantic_fallback:
        print("Using semantic similarity fallback...")
        semantic_KG = create_semantic_fallback_graph(class_list)
        
        print(f"Semantic fallback: {semantic_KG.number_of_nodes()} nodes, {semantic_KG.number_of_edges()} edges")
        
        # Merge graphs (ConceptNet edges take precedence)
        for u, v, data in semantic_KG.edges(data=True):
            if not KG.has_edge(u, v):
                KG.add_edge(u, v, **data)
    
    # Final graph statistics
    print(f"Final graph: {KG.number_of_nodes()} nodes, {KG.number_of_edges()} edges")
    
    return KG

def adjacency_matrix(G: nx.Graph, class_list: List[str]) -> torch.Tensor:
    """
    Enhanced adjacency matrix with better normalization
    """
    A = torch.zeros((len(class_list), len(class_list)))
    idx_map = {cls: i for i, cls in enumerate(class_list)}
    
    # Fill adjacency matrix
    for u, v, data in G.edges(data=True):
        if u in idx_map and v in idx_map:
            i, j = idx_map[u], idx_map[v]
            weight = data.get('weight', 1.0)
            A[i, j] = weight
            A[j, i] = weight
    
    # Add self-loops
    A += torch.eye(len(class_list))
    
    # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    degree = A.sum(1)
    # Handle isolated nodes (degree = 1 from self-loop)
    degree = torch.clamp(degree, min=1e-6)
    D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
    
    return D_inv_sqrt @ A @ D_inv_sqrt

def analyze_graph_connectivity(G: nx.Graph, class_list: List[str]) -> Dict:
    """
    Analyze graph connectivity and provide statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'num_classes': len(class_list),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G),
        'num_components': nx.number_connected_components(G),
        'isolated_nodes': list(nx.isolates(G)),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    }
    
    return stats

# Example usage and testing
if __name__ == "__main__":
    # Test with some sample classes
    test_classes = ['dog', 'cat', 'car', 'bicycle', 'person', 'chair', 'table', 'book']
    
    print("Testing robust knowledge graph utilities...")
    
    # Test regular build
    KG1 = build_knowledge_graph(test_classes)
    stats1 = analyze_graph_connectivity(KG1, test_classes)
    print(f"Regular build stats: {stats1}")
    
    # Test enhanced build
    KG2 = enhanced_build_knowledge_graph(test_classes)
    stats2 = analyze_graph_connectivity(KG2, test_classes)
    print(f"Enhanced build stats: {stats2}")
    
    # Test adjacency matrix
    adj = adjacency_matrix(KG2, test_classes)
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Adjacency matrix sum: {adj.sum().item():.2f}")