import os
import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent display errors
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np

from src.utils import load_metadata, get_or_create_embeddings
from src.recommender import get_complementary_recommendations
from src.catalog import create_unique_catalog
from src.search import reverse_product_search

def main():
    print("Initializing system for screenshot generation...")
    df_raw = load_metadata()
    # Sample 500 items to ensure we have duplicate clusters and accurate embeddings fast
    df_sampled, img_embs, txt_embs = get_or_create_embeddings(df_raw, sample_size=500)
    print(f"Sampled {len(df_sampled)} items successfully.")
    
    # ------------------ TASK 1 SCREENSHOT ------------------
    print("Generating Task 1 Recommendations screenshot...")
    # Find a sports shoe or fallback
    shoes = df_sampled[df_sampled['subCategory'] == 'Shoes']
    sports_shoes = shoes[shoes['usage'] == 'Sports'] if 'usage' in shoes.columns else shoes
    sample_shoe = sports_shoes.iloc[0] if len(sports_shoes) > 0 else df_sampled.iloc[0]
    
    recs = get_complementary_recommendations(sample_shoe['id'], df_sampled, img_embs, top_k=3)
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    # Query product
    img_q = Image.open(sample_shoe['image_path'])
    axes[0].imshow(img_q)
    axes[0].axis('off')
    axes[0].set_title(f"INPUT PRODUCT\n{sample_shoe['productDisplayName'][:20]}...", fontsize=10, fontweight='bold', color='blue')
    
    # Recommendations
    for idx, rec in enumerate(recs):
        img_rec = Image.open(rec['image_path'])
        ax = axes[idx + 1]
        ax.imshow(img_rec)
        ax.axis('off')
        ax.set_title(f"{rec['relationship']}\nScore: {rec['score']:.2f}\n{rec['productDisplayName'][:20]}...", fontsize=9)
        
    plt.tight_layout()
    plt.savefig('task1_recommendation_results.png', dpi=150)
    plt.close()
    print("Saved task1_recommendation_results.png")
    
    # ------------------ TASK 2 SCREENSHOT ------------------
    print("Generating Task 2 Deduplication screenshot...")
    unique_df, mapping, duplicates, stats = create_unique_catalog(df_sampled, img_embs, similarity_threshold=0.85)
    
    # Find a duplicate group with at least 2 items
    target_cluster = None
    for dup in duplicates:
        if len(dup['items']) >= 2:
            target_cluster = dup
            break
            
    if target_cluster:
        items = target_cluster['items']
        num_items = min(5, len(items))
        fig, axes = plt.subplots(1, num_items, figsize=(3 * num_items, 4))
        if num_items == 1:
            axes = [axes]
            
        for i in range(num_items):
            item = items[i]
            img = Image.open(item['image_path'])
            ax = axes[i]
            ax.imshow(img)
            ax.axis('off')
            if item['id'] == target_cluster['representative_id']:
                ax.set_title(f"REPRESENTATIVE\nID: {item['id']}\n{item['productDisplayName'][:15]}...", fontsize=9, fontweight='bold', color='green')
            else:
                ax.set_title(f"DUPLICATE ITEM\nID: {item['id']}\n{item['productDisplayName'][:15]}...", fontsize=9, color='red')
        plt.tight_layout()
        plt.savefig('task2_deduplication_results.png', dpi=150)
        plt.close()
        print("Saved task2_deduplication_results.png")
    else:
        # Fallback empty or dummy plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No duplicates found in tiny sample", ha='center', va='center')
        plt.savefig('task2_deduplication_results.png', dpi=150)
        plt.close()
        print("Saved dummy task2_deduplication_results.png")
        
    # ------------------ TASK 3 SCREENSHOT ------------------
    print("Generating Task 3 Search screenshot...")
    query = "blue shirt"
    results = reverse_product_search(query, df_sampled, img_embs, top_k=4)
    
    fig, axes = plt.subplots(1, len(results), figsize=(14, 4))
    if len(results) == 1:
        axes = [axes]
        
    for idx, res in enumerate(results):
        img = Image.open(res['image_path'])
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Match: {res['score']*100:.1f}%\n{res['productDisplayName'][:20]}...", fontsize=9)
        
    plt.suptitle(f"Reverse Search Results for: '{query}'", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('task3_search_results.png', dpi=150)
    plt.close()
    print("Saved task3_search_results.png")
    
    print("All screenshots generated successfully!")

if __name__ == "__main__":
    main()
