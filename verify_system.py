import sys
import os

try:
    print("--- 1. Testing Module Imports ---")
    from src.utils import load_metadata, get_or_create_embeddings
    from src.recommender import get_complementary_recommendations
    from src.catalog import create_unique_catalog
    from src.search import reverse_product_search
    print("Imports OK!")
    
    print("\n--- 2. Loading Dataset Metadata ---")
    df = load_metadata()
    print(f"Dataset loaded: {len(df)} items with images.")
    
    print("\n--- 3. Running Embedding Extraction (Tiny Sample for Verification) ---")
    # Use sample_size=100 for verification to run very quickly
    df_sampled, img_embs, txt_embs = get_or_create_embeddings(df, sample_size=100, force_recompute=True)
    print(f"Sampled {len(df_sampled)} items.")
    print(f"Image embeddings shape: {img_embs.shape}")
    print(f"Text embeddings shape: {txt_embs.shape}")
    
    print("\n--- 4. Testing Task 1: Smart Recommendations ---")
    sample_id = df_sampled.iloc[0]['id']
    print(f"Input Product ID: {sample_id} ({df_sampled.iloc[0]['productDisplayName']})")
    recs = get_complementary_recommendations(sample_id, df_sampled, img_embs, top_k=2)
    print(f"Generated {len(recs)} recommendations.")
    for idx, rec in enumerate(recs):
        print(f"  Rec {idx+1}: {rec['productDisplayName']} | Relation: {rec['relationship']} | Score: {rec['score']:.4f}")
        
    print("\n--- 5. Testing Task 2: Catalog Deduplication ---")
    unique_df, mapping, duplicates, stats = create_unique_catalog(df_sampled, img_embs, similarity_threshold=0.85)
    print(f"Deduplication completed:")
    print(f"  Original Count: {stats['total_original_products']}")
    print(f"  Unique Count: {stats['total_unique_products']}")
    print(f"  Duplicate Groups: {stats['duplicate_groups_found']}")
    print(f"  Compression Ratio: {stats['compression_percentage']}%")
    
    print("\n--- 6. Testing Task 3: Reverse Product Search ---")
    query = "blue shirt"
    print(f"Query: '{query}'")
    results = reverse_product_search(query, df_sampled, img_embs, top_k=3)
    print(f"Retrieved {len(results)} matches:")
    for idx, res in enumerate(results):
        print(f"  Match {idx+1}: {res['productDisplayName']} | Match Score: {res['score']*100:.1f}%")
        
    print("\n=============================================")
    print("[SUCCESS] ALL TESTS PASSED AND SYSTEM VERIFIED!")
    print("=============================================")
    sys.exit(0)
    
except Exception as e:
    import traceback
    print("\n[ERROR] Verification failed with error:")
    traceback.print_exc()
    sys.exit(1)
