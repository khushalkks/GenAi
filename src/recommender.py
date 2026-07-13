import numpy as np
import pandas as pd

def get_complementary_recommendations(product_id, df, image_embeddings, top_k=3):
    """
    Suggests complementary products for a given product ID.
    
    Complementary logic:
    - Shoes/Footwear -> Socks, Watches, Sports Apparel (if Sports) or Bottomwear (if Casual)
    - Topwear -> Bottomwear, Shoes, Watches/Belts
    - Bottomwear -> Topwear, Shoes, Belts
    - Watches -> Apparel, Shoes, Bags
    - Bags/Wallets -> Apparel, Shoes, Watches
    
    Filters candidates by gender and usage, then ranks by CLIP visual embedding similarity.
    """
    if product_id not in df['id'].values:
        raise ValueError(f"Product ID {product_id} not found in database.")
        
    query_idx = df[df['id'] == product_id].index[0]
    query_row = df.iloc[query_idx]
    
    master_cat = query_row['masterCategory']
    sub_cat = query_row['subCategory']
    art_type = query_row['articleType']
    gender = query_row['gender']
    usage = query_row['usage']
    
    # 1. Determine target complementary subCategories/articleTypes
    target_categories = []
    
    if sub_cat == 'Shoes' or master_cat == 'Footwear':
        target_categories = [
            {'subCategory': 'Socks'},
            {'subCategory': 'Watches'},
            {'masterCategory': 'Apparel'}  # e.g., Tshirts or Shorts
        ]
    elif sub_cat == 'Topwear' or art_type in ['Tshirts', 'Shirts']:
        target_categories = [
            {'subCategory': 'Bottomwear'},
            {'subCategory': 'Shoes'},
            {'subCategory': 'Watches'}
        ]
    elif sub_cat == 'Bottomwear':
        target_categories = [
            {'subCategory': 'Topwear'},
            {'subCategory': 'Shoes'},
            {'subCategory': 'Belts'}
        ]
    elif sub_cat == 'Watches':
        target_categories = [
            {'masterCategory': 'Apparel'},
            {'subCategory': 'Shoes'},
            {'subCategory': 'Bags'}
        ]
    elif sub_cat in ['Bags', 'Wallets']:
        target_categories = [
            {'masterCategory': 'Apparel'},
            {'subCategory': 'Shoes'},
            {'subCategory': 'Watches'}
        ]
    else:
        # Default fallback: pick other subcategories under the same master category,
        # or cross-category if not possible.
        all_subcats = [s for s in df['subCategory'].unique() if s != sub_cat]
        target_categories = [
            {'subCategory': all_subcats[0]} if len(all_subcats) > 0 else {'masterCategory': 'Accessories'},
            {'subCategory': all_subcats[1]} if len(all_subcats) > 1 else {'masterCategory': 'Footwear'},
            {'subCategory': all_subcats[2]} if len(all_subcats) > 2 else {'masterCategory': 'Apparel'}
        ]
        
    # 2. Score candidates for each target category
    query_emb = image_embeddings[query_idx]
    recommendations = []
    
    for target in target_categories[:top_k]:
        candidates_df = df.copy()
        
        # Apply category/subcategory filters
        for col, val in target.items():
            candidates_df = candidates_df[candidates_df[col] == val]
            
        # Exclude the query product itself
        candidates_df = candidates_df[candidates_df['id'] != product_id]
        
        # Fallback if no products match the exact subcategory
        if len(candidates_df) == 0:
            candidates_df = df[df['id'] != product_id]
            if 'masterCategory' in target:
                candidates_df = candidates_df[candidates_df['masterCategory'] == target['masterCategory']]
            elif 'subCategory' in target:
                # Find something in same masterCategory but different subCategory
                candidates_df = candidates_df[
                    (candidates_df['masterCategory'] == master_cat) & 
                    (candidates_df['subCategory'] != sub_cat)
                ]
                
        if len(candidates_df) == 0:
            continue
            
        # Filter by gender compatibility:
        # E.g., Men's product -> Men or Unisex candidates.
        # Women's product -> Women or Unisex candidates.
        gender_filtered = candidates_df[candidates_df['gender'].isin([gender, 'Unisex'])]
        if len(gender_filtered) > 0:
            candidates_df = gender_filtered
            
        # Filter by usage alignment (Casual, Sports, Formal, etc.)
        if pd.notna(usage):
            usage_filtered = candidates_df[candidates_df['usage'] == usage]
            if len(usage_filtered) > 0:
                candidates_df = usage_filtered
                
        # Find candidate indexes in the parent dataframe
        candidate_indices = candidates_df.index.tolist()
        
        # Calculate cosine similarity with the query image embedding
        candidate_embs = image_embeddings[candidate_indices]
        
        # Compute dot products divided by norms
        dot_product = np.dot(candidate_embs, query_emb)
        candidate_norms = np.linalg.norm(candidate_embs, axis=1)
        query_norm = np.linalg.norm(query_emb)
        
        similarities = dot_product / (candidate_norms * query_norm + 1e-8)
        
        # Get the top matching item in this target category
        best_idx_in_candidates = np.argmax(similarities)
        best_global_idx = candidate_indices[best_idx_in_candidates]
        score = similarities[best_idx_in_candidates]
        best_product = df.iloc[best_global_idx]
        
        # Determine readable relationship name
        rel_name = target.get('subCategory') or target.get('masterCategory')
        
        recommendations.append({
            'id': best_product['id'],
            'gender': best_product['gender'],
            'masterCategory': best_product['masterCategory'],
            'subCategory': best_product['subCategory'],
            'articleType': best_product['articleType'],
            'baseColour': best_product['baseColour'],
            'usage': best_product['usage'],
            'productDisplayName': best_product['productDisplayName'],
            'image_path': best_product['image_path'],
            'score': float(score),
            'relationship': f"Complementary {rel_name}"
        })
        
    return recommendations
