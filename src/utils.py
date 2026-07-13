import os
import pandas as pd
import numpy as np
import pickle
import torch
from PIL import Image
import kagglehub
from sentence_transformers import SentenceTransformer

def get_dataset_path():
    """Retrieve dataset path programmatically using kagglehub."""
    try:
        return kagglehub.dataset_download('paramaggarwal/fashion-product-images-small')
    except Exception as e:
        workspace_path = os.getcwd()
        if os.path.exists(os.path.join(workspace_path, "styles.csv")):
            return workspace_path
        raise RuntimeError(f"Failed to find dataset: {e}")

def load_metadata():
    """Load metadata from styles.csv and filter by existing images."""
    dataset_dir = get_dataset_path()
    csv_path = os.path.join(dataset_dir, "styles.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file styles.csv not found at {csv_path}")
        
    # styles.csv has some malformed lines with bad number of fields.
    # We skip them robustly using on_bad_lines='skip'.
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Ensure IDs are strings
    df['id'] = df['id'].astype(str)
    
    # Resolve images folder path
    images_dir = os.path.join(dataset_dir, "images")
    df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
    df['image_exists'] = df['image_path'].apply(os.path.exists)
    
    # Filter only rows with valid image files
    df = df[df['image_exists']].reset_index(drop=True)
    return df

_model = None

def get_clip_model():
    """Load and cache the CLIP model from sentence-transformers."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model 'clip-ViT-B-32' on device: {device}...")
        _model = SentenceTransformer('clip-ViT-B-32', device=device)
    return _model

def get_or_create_embeddings(df, sample_size=3000, force_recompute=False):
    """Retrieve pre-computed embeddings if cached, else compute and cache them."""
    embeddings_file = os.path.join(os.getcwd(), "embeddings_cache.pkl")
    
    if os.path.exists(embeddings_file) and not force_recompute:
        print("Loading embeddings from cache...")
        try:
            with open(embeddings_file, 'rb') as f:
                cache = pickle.load(f)
            cached_ids = cache.get('ids', [])
            if len(cached_ids) > 0 and all(cid in df['id'].values for cid in cached_ids):
                # Filter and align dataframe with cached IDs
                df_sampled = df[df['id'].isin(cached_ids)].set_index('id').loc[cached_ids].reset_index()
                print(f"Successfully loaded {len(df_sampled)} cached product embeddings.")
                return df_sampled, cache['image_embeddings'], cache['text_embeddings']
        except Exception as e:
            print(f"Cache load failed ({e}). Recomputing embeddings...")
            
    # Generate embeddings
    print("Generating embeddings. This may take a few minutes...")
    
    # To maintain category diversity, sample across subCategories
    if sample_size is not None and len(df) > sample_size:
        # Group by subCategory and sample proportionally
        subcat_counts = df['subCategory'].value_counts()
        # Ensure we don't crash on very small categories
        df_sampled = df.groupby('subCategory', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df)))), random_state=42)
        )
        # Adjust sample size if not exact
        if len(df_sampled) > sample_size:
            df_sampled = df_sampled.sample(sample_size, random_state=42)
        elif len(df_sampled) < sample_size:
            remaining = df[~df['id'].isin(df_sampled['id'])]
            df_sampled = pd.concat([df_sampled, remaining.sample(sample_size - len(df_sampled), random_state=42)])
        df_sampled = df_sampled.reset_index(drop=True)
    else:
        df_sampled = df.copy()
        
    model = get_clip_model()
    
    # Process images
    image_paths = df_sampled['image_path'].tolist()
    images = []
    valid_indices = []
    
    print(f"Loading {len(image_paths)} images...")
    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            pass # Skip missing or corrupt images
            
    # Align dataframe with loaded images
    df_sampled = df_sampled.iloc[valid_indices].reset_index(drop=True)
    
    # Encode images
    print(f"Encoding {len(images)} product images with CLIP...")
    image_embeddings = model.encode(images, batch_size=128, show_progress_bar=True)
    
    # Generate search text descriptions
    df_sampled['search_text'] = df_sampled.apply(
        lambda r: f"{r['gender']} {r['baseColour']} {r['articleType']} {r['productDisplayName']}", axis=1
    )
    texts = df_sampled['search_text'].tolist()
    
    # Encode text descriptions
    print(f"Encoding {len(texts)} product descriptions with CLIP...")
    text_embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
    
    # Cache to file
    cache = {
        'ids': df_sampled['id'].tolist(),
        'image_embeddings': image_embeddings,
        'text_embeddings': text_embeddings
    }
    with open(embeddings_file, 'wb') as f:
        pickle.dump(cache, f)
        
    print(f"Embeddings saved to cache: {embeddings_file}")
    return df_sampled, image_embeddings, text_embeddings
