import numpy as np
import pandas as pd
from src.utils import get_clip_model

def reverse_product_search(query_text, df, image_embeddings, top_k=5):
    """
    Performs text-based search over product images using CLIP cross-modal embeddings.
    
    Returns:
    - results: List of top_k matching products with similarity scores.
    """
    if not query_text.strip():
        return []
        
    model = get_clip_model()
    
    # 1. Encode text query using CLIP
    print(f"Encoding text query: '{query_text}'...")
    query_embedding = model.encode(query_text, show_progress_bar=False)
    
    # 2. Normalize embeddings for cosine similarity
    query_norm = np.linalg.norm(query_embedding)
    query_emb_normalized = query_embedding / (query_norm + 1e-8)
    
    img_norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    img_embs_normalized = image_embeddings / (img_norms + 1e-8)
    
    # 3. Compute cosine similarity
    similarities = np.dot(img_embs_normalized, query_emb_normalized)
    
    # 4. Get the indices of the top-k highest similarity products
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        product = df.iloc[idx]
        score = similarities[idx]
        
        results.append({
            'id': product['id'],
            'gender': product['gender'],
            'masterCategory': product['masterCategory'],
            'subCategory': product['subCategory'],
            'articleType': product['articleType'],
            'baseColour': product['baseColour'],
            'usage': product['usage'],
            'productDisplayName': product['productDisplayName'],
            'image_path': product['image_path'],
            'score': float(score)
        })
        
    return results
