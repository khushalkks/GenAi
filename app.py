import streamlit as st
import os
import random
import numpy as np
from PIL import Image
import pandas as pd

# Set page config with title, icon, and layout
st.set_page_config(
    page_title="AI Product Intelligence System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Apply font to elements */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Title styling */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a8a5e6, #6c5ce7, #00d2d3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    text-align: center;
}

.sub-title {
    font-size: 1.2rem;
    color: #7f8c8d;
    text-align: center;
    margin-bottom: 2.5rem;
}

/* Custom cards */
.card {
    background-color: #1e1e2f;
    border: 1px solid #2d2d44;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    margin-bottom: 1.5rem;
}

.product-card {
    background-color: #151522;
    border: 1px solid #252538;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    height: 100%;
}

.product-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(108, 92, 231, 0.25);
    border-color: #6c5ce7;
}

.product-title {
    font-size: 0.95rem;
    font-weight: 600;
    margin: 8px 0 4px 0;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    height: 40px;
}

.product-meta {
    font-size: 0.8rem;
    color: #888;
}

.relationship-badge {
    background-color: #6c5ce7;
    color: white;
    font-size: 0.75rem;
    font-weight: bold;
    padding: 3px 8px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 8px;
}

/* Stat banner metrics */
.metric-box {
    background: linear-gradient(135deg, #2a2a40, #1b1b2a);
    border: 1px solid #3d3d5c;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d2d3;
    margin: 5px 0;
}

.metric-label {
    font-size: 0.9rem;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# Imports from custom modules
try:
    from src.utils import load_metadata, get_or_create_embeddings
    from src.recommender import get_complementary_recommendations
    from src.catalog import create_unique_catalog
    from src.search import reverse_product_search
except ImportError as e:
    st.error(f"Failed to import local modules: {e}")
    st.info("Make sure the application is run from the workspace root.")

# Main Application Banner
st.markdown('<div class="main-title">Product Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Gen AI Bootcamp - Day 2 Homework Assignment</div>', unsafe_allow_html=True)

# Load dataset and cache it
@st.cache_resource
def init_system():
    with st.spinner("Initializing system and loading dataset..."):
        df_raw = load_metadata()
        # Default sample to 3000 for responsive runtime
        df, img_embs, txt_embs = get_or_create_embeddings(df_raw, sample_size=3000)
    return df, img_embs, txt_embs

try:
    df, img_embs, txt_embs = init_system()
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.info("Verify your internet connection for the initial dataset download.")
    st.stop()

# Sidebar Setup
st.sidebar.image("https://img.icons8.com/clouds/200/shopping-bag.png", width=120)
st.sidebar.title("Configuration")
st.sidebar.markdown("Configure hyperparameters for recommendations, text queries, and clustering.")

# App Mode Selector
app_tab = st.sidebar.radio(
    "Select Task",
    ["Task 1: Recommendation Engine", "Task 2: Unique Catalog", "Task 3: Reverse Search"]
)

# Sidebar helper info
st.sidebar.divider()
st.sidebar.subheader("System Information")
st.sidebar.info(f"Loaded Products: **{len(df)}**\nEmbeddings cached locally.")

# ----------------- TASK 1: SMART RECOMMENDATION ENGINE -----------------
if app_tab == "Task 1: Recommendation Engine":
    st.header("Task 1: Smart Product Recommendation Engine")
    st.write("Find complementary products commonly purchased together (e.g. shoes suggest socks, watches, apparel).")
    
    st.divider()
    
    # Let user pick an item
    col_input, col_space = st.columns([2, 3])
    with col_input:
        subcats = sorted(df['subCategory'].unique())
        selected_subcat = st.selectbox("Filter products by Subcategory", subcats, index=subcats.index("Shoes") if "Shoes" in subcats else 0)
        
        # Filter products in subcategory
        subcat_df = df[df['subCategory'] == selected_subcat]
        prod_names = subcat_df['productDisplayName'].tolist()
        prod_ids = subcat_df['id'].tolist()
        
        selected_name = st.selectbox(
            "Select Input Product",
            prod_names,
            index=0 if prod_names else 0
        )
        
        query_idx = prod_names.index(selected_name)
        selected_id = prod_ids[query_idx]
        
    st.subheader("Selected Item & Smart Complementary Recommendations")
    
    col_prod, col_recs = st.columns([1.5, 3.5])
    
    # Query product details
    q_product = df[df['id'] == selected_id].iloc[0]
    
    with col_prod:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("##### Input Product Details")
        st.image(q_product['image_path'], use_container_width=True)
        st.write(f"**Name:** {q_product['productDisplayName']}")
        st.write(f"**Category:** {q_product['masterCategory']} / {q_product['subCategory']}")
        st.write(f"**Usage:** {q_product['usage']}")
        st.write(f"**Color:** {q_product['baseColour']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_recs:
        try:
            recs = get_complementary_recommendations(selected_id, df, img_embs, top_k=3)
            
            if not recs:
                st.info("No complementary products found.")
            else:
                rec_cols = st.columns(len(recs))
                for idx, col_cell in enumerate(rec_cols):
                    rec = recs[idx]
                    with col_cell:
                        st.markdown(f"""
                        <div class="product-card">
                            <div class="relationship-badge">{rec['relationship']}</div>
                            <img src="app/static/{rec['image_path']}" style="width:100%; border-radius:8px; margin-bottom:8px;" onerror="this.src='{rec['image_path']}';">
                            <div class="product-title" title="{rec['productDisplayName']}">{rec['productDisplayName']}</div>
                            <div class="product-meta">
                                <b>Match Score:</b> {rec['score']:.2f}<br>
                                <b>Gender:</b> {rec['gender']}<br>
                                <b>Color:</b> {rec['baseColour']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error fetching recommendations: {e}")

# ----------------- TASK 2: UNIQUE PRODUCT CATALOG -----------------
elif app_tab == "Task 2: Unique Catalog":
    st.header("Task 2: Unique Product Catalog Creation")
    st.write("Group duplicate/near-duplicate products based on image similarity to produce a clean catalog of unique products.")
    
    # Adjust threshold in sidebar
    st.sidebar.subheader("Clustering Parameters")
    thresh = st.sidebar.slider(
        "Cosine Similarity Threshold",
        min_value=0.70,
        max_value=0.98,
        value=0.88,
        step=0.01,
        help="Higher values create tighter clusters of duplicate items."
    )
    
    with st.spinner("Generating unique catalog by clustering embeddings..."):
        unique_df, mapping, duplicates, stats = create_unique_catalog(df, img_embs, similarity_threshold=thresh)
        
    st.divider()
    
    # Styled metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{stats["total_original_products"]}</div><div class="metric-label">Original Items</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{stats["total_unique_products"]}</div><div class="metric-label">Unique Catalog Items</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{stats["duplicate_groups_found"]}</div><div class="metric-label">Duplicate Groups</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{stats["compression_percentage"]}%</div><div class="metric-label">Redundancy Removed</div></div>', unsafe_allow_html=True)
        
    st.divider()
    
    col_dup, col_cat = st.columns([2.5, 2.5])
    
    with col_dup:
        st.subheader("Browse Duplicate Groups Detected")
        st.write("Markets often contain identical/near-identical listings. Here are the clusters grouped by the algorithm:")
        
        if not duplicates:
            st.info("No duplicate groups found. Try lowering the Cosine Similarity Threshold in the sidebar.")
        else:
            selected_cluster_idx = st.selectbox(
                "Select Duplicate Cluster to Inspect",
                range(len(duplicates)),
                format_func=lambda idx: f"Cluster {idx+1} ({len(duplicates[idx]['items'])} items): {duplicates[idx]['representative_name']}"
            )
            
            cluster = duplicates[selected_cluster_idx]
            
            st.markdown(f"**Representative Item in Clean Catalog:** `{cluster['representative_name']}` (ID: {cluster['representative_id']})")
            
            st.write("##### Items in this Duplicate Group:")
            
            cols = st.columns(min(4, len(cluster['items'])))
            for idx, item in enumerate(cluster['items']):
                with cols[idx % 4]:
                    st.markdown(f"""
                    <div class="product-card">
                        <img src="app/static/{item['image_path']}" style="width:100%; border-radius:8px; margin-bottom:8px;" onerror="this.src='{item['image_path']}';">
                        <div class="product-title" title="{item['productDisplayName']}">{item['productDisplayName']}</div>
                        <div class="product-meta">
                            <b>ID:</b> {item['id']}<br>
                            <b>Color:</b> {item['baseColour']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    with col_cat:
        st.subheader("Final Cleaned Unique Catalog Preview")
        st.write("The deduplicated catalog with standardized names:")
        
        preview_cols = ['id', 'canonicalDisplayName', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour']
        st.dataframe(
            unique_df[preview_cols].rename(columns={'canonicalDisplayName': 'productDisplayName'}),
            use_container_width=True,
            height=380
        )

# ----------------- TASK 3: REVERSE PRODUCT SEARCH -----------------
elif app_tab == "Task 3: Reverse Search":
    st.header("Task 3: Reverse Product Search")
    st.write("Search the catalog visually using text queries by matching CLIP text embeddings against product image embeddings.")
    
    st.divider()
    
    # Search box inputs
    col_search, col_k = st.columns([3.5, 1.5])
    with col_search:
        search_query = st.text_input(
            "Enter text search query",
            placeholder="e.g. blue casual shirt, black sporty running shoe, red print dress, leather bag"
        )
    with col_k:
        k_results = st.slider("Number of results to display", min_value=1, max_value=12, value=4)
        
    st.subheader("Top Matching Products")
    
    if search_query:
        with st.spinner(f"Searching for '{search_query}'..."):
            results = reverse_product_search(search_query, df, img_embs, top_k=k_results)
            
        if not results:
            st.warning("No matching products found.")
        else:
            cols = st.columns(min(4, len(results)))
            for idx, res in enumerate(results):
                with cols[idx % 4]:
                    score_pct = res['score'] * 100
                    st.markdown(f"""
                    <div class="product-card">
                        <div class="relationship-badge" style="background-color: #10ac84;">Match: {score_pct:.1f}%</div>
                        <img src="app/static/{res['image_path']}" style="width:100%; border-radius:8px; margin-bottom:8px;" onerror="this.src='{res['image_path']}';">
                        <div class="product-title" title="{res['productDisplayName']}">{res['productDisplayName']}</div>
                        <div class="product-meta">
                            <b>Subcategory:</b> {res['subCategory']}<br>
                            <b>Gender:</b> {res['gender']}<br>
                            <b>Usage:</b> {res['usage']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Type a query in the search bar above to query the catalog.")
        
        # Display some sample queries
        st.write("##### Try these sample searches:")
        sample_queries = [
            "blue casual shirt", 
            "red party dress", 
            "running shoe", 
            "sports socks", 
            "black leather belt", 
            "sport watch"
        ]
        
        cols = st.columns(len(sample_queries))
        for idx, query in enumerate(sample_queries):
            with cols[idx]:
                if st.button(query, use_container_width=True):
                    # We trigger a rerun by setting query in st.session_state
                    st.session_state.search_query_click = query
                    
        # Handle button click query injection
        if 'search_query_click' in st.session_state:
            st.info(f"Click search bar and hit Enter to search: **{st.session_state.search_query_click}**")
