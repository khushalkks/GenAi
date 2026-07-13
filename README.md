# 🛍️ AI Product Intelligence System
### A Multimodal Product Discovery & Catalog Deduplication Engine

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-yellow?style=for-the-badge)](https://huggingface.co)

This repository contains the complete implementation of the **AI Product Intelligence System** built using pre-trained **CLIP (Contrastive Language-Image Pretraining)** embeddings, vector search, and custom algorithmic heuristics.

---

## 🎯 Project Overview & Core Tasks

### 📈 Task 1: Smart Product Recommendation Engine
Standard recommendation engines search for *visually identical* products. To improve user conversion and cross-selling, we engineered a system to recommend **complementary** items (e.g., matching running shoes with socks, apparel, and a fitness watch).

*   **Rule-Based Category Linkages:** Maps search categories directly to coordinate styled accessories (e.g., `Footwear` $\rightarrow$ `Socks`, `Watches`, `Apparel`).
*   **Contextual Filtering:** Restricts candidate recommendations by `gender` (e.g., Men or Unisex for male items) and `usage` (locking suggestions to `Sports` accessories when viewing running shoes).
*   **CLIP Visual Affinity:** Ranks candidates within coordinate categories using **cosine similarity** of CLIP image embeddings.

#### 📊 Task 1 Visual Results
![Task 1 Recommendations Results](task1_recommendation_results.png)

---

### 🗃️ Task 2: Unique Product Catalog Creation
Marketplaces face data quality issues due to sellers uploading near-duplicate products (e.g., slightly altered names, different packages like "Pack of 2"). This pipeline automatically groups duplicates and cleanses the database into a canonical unique catalog.

*   **Leader-Centroid Clustering:** Designates the first unvisited item as a leader and clusters all other unvisited items with cosine similarity $\ge 0.88$ (adjustable threshold).
*   **Medoid Selection:** To select a single "canonical" product for each cluster, it computes the pairwise intra-cluster similarity matrix and identifies the medoid item (highest average similarity).
*   **Canonical Name Synthesis:** Standardizes titles by cleaning noise suffixes, duplicate indices, pack counts (e.g. "Pack of 3", "Size L", "Combo", "Shoe A") using Regex patterns.

#### 📊 Task 2 Visual Results
![Task 2 Deduplication Results](task2_deduplication_results.png)

---

### 🔍 Task 3: Reverse Product Search
Enables natural language text search against product image databases by mapping textual semantics directly to visual assets in a shared vector space.

*   **Shared Space Retrieval:** Maps text queries into the same joint embedding space as product images using the CLIP text encoder.
*   **Semantic Matching:** Ranks candidate images by computing cosine similarity between the query text embedding and image embeddings.

#### 📊 Task 3 Visual Results
![Task 3 Search Results](task3_search_results.png)

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/khushalkks/GenAi.git
    cd GenAi
    ```

2.  **Install Required Libraries:**
    ```bash
    python -m pip install streamlit sentence-transformers pandas numpy pillow scikit-learn matplotlib kagglehub
    ```

3.  **Run the Interactive Streamlit Web Application:**
    ```bash
    streamlit run app.py
    ```
    This launches a browser dashboard at `http://localhost:8501/`.

4.  **Run the Self-Contained Notebook:**
    *   Open `Bootcamp_Day2_Homework.ipynb` in VS Code or Jupyter Lab.
    *   Run cells sequentially to download the Kaggle dataset programmatically and execute verification.
