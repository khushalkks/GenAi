import numpy as np
import pandas as pd
import re

def clean_name(name):
    """
    Cleans product display names by removing noise words, sizes, 
    and duplicate indicators (e.g. 'Pack of 2', 'Combo', 'A', 'B', '1', '2').
    """
    if not isinstance(name, str):
        return ""
    
    # Remove things like "Pack of X", "Combo of X", "Set of X", "Pair of X"
    name = re.sub(r'(?i)\b(pack|combo|set|pair|size|width)\b.*', '', name)
    
    # Remove trailing single letters or digits preceded by spaces (e.g., "Shoe A", "Shirt 1")
    name = re.sub(r'\s+[A-Za-z0-9]$', '', name)
    
    # Remove special characters and clean extra whitespaces
    name = re.sub(r'[-_/,\(\)]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def create_unique_catalog(df, image_embeddings, similarity_threshold=0.88):
    """
    Groups duplicate or near-duplicate items using leader-based clustering 
    on CLIP image embeddings, selects a medoid representative for each cluster,
    and returns a cleaned deduplicated catalog along with cluster details.
    
    Returns:
    - unique_catalog_df: DataFrame of unique products.
    - cluster_mapping: Dictionary mapping duplicate IDs to their representative ID.
    - duplicate_clusters: List of lists containing product details of duplicate groups.
    - stats: Dictionary of catalog statistics.
    """
    n = len(df)
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    normalized_embs = image_embeddings / (norms + 1e-8)
    
    visited = np.zeros(n, dtype=bool)
    clusters = []
    
    # Leader clustering (Single-pass clustering around centroid leaders)
    for i in range(n):
        if visited[i]:
            continue
            
        # Compute cosine similarity of leader i to all products
        sims = np.dot(normalized_embs, normalized_embs[i])
        
        # Find all unvisited products with similarity above threshold
        cluster_indices = np.where((sims >= similarity_threshold) & (~visited))[0]
        
        # Mark them as visited
        visited[cluster_indices] = True
        
        clusters.append(cluster_indices.tolist())
        
    unique_products = []
    cluster_mapping = {}
    duplicate_clusters = []
    
    for cluster in clusters:
        # If a cluster has multiple items, find the medoid (representative item)
        if len(cluster) > 1:
            cluster_embs = normalized_embs[cluster]
            # Compute pairwise similarities within cluster
            cluster_sims = np.dot(cluster_embs, cluster_embs.T)
            # Medoid is the one with the highest sum of similarities to others
            medoid_idx_in_cluster = np.argmax(cluster_sims.sum(axis=1))
            representative_idx = cluster[medoid_idx_in_cluster]
        else:
            representative_idx = cluster[0]
            
        rep_product = df.iloc[representative_idx].copy()
        
        # Apply name cleaning to the representative product
        rep_product['canonicalDisplayName'] = clean_name(rep_product['productDisplayName'])
        
        # If canonical name is too short or empty, fallback
        if len(rep_product['canonicalDisplayName']) < 3:
            rep_product['canonicalDisplayName'] = rep_product['productDisplayName']
            
        unique_products.append(rep_product)
        
        # Track duplicate mappings and compile details for clusters with duplicates
        rep_id = rep_product['id']
        cluster_items = []
        
        for idx in cluster:
            item = df.iloc[idx]
            cluster_mapping[item['id']] = rep_id
            
            cluster_items.append({
                'id': item['id'],
                'productDisplayName': item['productDisplayName'],
                'articleType': item['articleType'],
                'baseColour': item['baseColour'],
                'image_path': item['image_path']
            })
            
        if len(cluster) > 1:
            duplicate_clusters.append({
                'representative_id': rep_id,
                'representative_name': rep_product['canonicalDisplayName'],
                'items': cluster_items
            })
            
    # Build unique catalog dataframe
    unique_catalog_df = pd.DataFrame(unique_products).reset_index(drop=True)
    
    # Sort duplicate clusters by size (largest groups of duplicates first)
    duplicate_clusters = sorted(duplicate_clusters, key=lambda c: len(c['items']), reverse=True)
    
    # Calculate statistics
    stats = {
        'total_original_products': n,
        'total_unique_products': len(unique_catalog_df),
        'duplicate_groups_found': len(duplicate_clusters),
        'duplicates_removed': n - len(unique_catalog_df),
        'compression_percentage': round(((n - len(unique_catalog_df)) / n) * 100, 2)
    }
    
    return unique_catalog_df, cluster_mapping, duplicate_clusters, stats
