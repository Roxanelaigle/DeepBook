# 📚 CamemBERT-KNN Book Recommendation System

This repository implements a **book recommendation system** using **CamemBERT embeddings** and a **hybrid search approach** (cosine similarity or k-Nearest Neighbors). The system incorporates **genre embeddings** and a **curiosity gauge** to fine-tune recommendations.

## Features
- **Uses embeddings** for book titles, descriptions, and optionally, genres.
- **Supports two recommendation methods**:
  - **Cosine Similarity**: Measures the similarity between books.
  - **k-Nearest Neighbors (KNN)**: Finds the most similar books in the dataset.
- **Curiosity Gauge**: Allows exploration of results beyond top-ranked books.
- **Weighted Combination (`alpha`)**: Balances importance between title/description and genre embeddings.

---

## 📚 How the Recommendation Algorithm Works

### **1️. Embedding Generation**
Each book is converted into **vector representations** (embeddings) using **CamemBERT**:
- **Title + Description** → `titledesc` embeddings
- **Genre** → `genre` embeddings (if enabled)

For an input book:
- `input_embedding`: Generated from **title and description**.
- `input_genre_embedding`: Generated from **book genre** (if applicable).

### **2. Computing Similarities**
Two methods are available:

#### **A) Cosine Similarity**
- We compute **separate similarities**:

```math
similarity_{title} = \cos(dataset\_embedding_{titledesc}, input\_embedding_{titledesc})
```

---

```math
similarity_{genre} = \cos(dataset\_embedding_{genre}, input\_embedding_{genre})
```

- The final similarity score is a **weighted combination**:

```math
similarity = \alpha \cdot similarity_{titledesc} + (1 - \alpha) \cdot similarity_{genre} $$
```

- Books are ranked based on **1 - similarity** (lower is better).

```math
similarity = \alpha \cdot similarity_{titledesc} + (1 - \alpha) \cdot similarity_{genre}
```

#### **B) k-Nearest Neighbors (KNN)**
- We train separate **KNN models** for **title/description** and **genre**.
- Each model returns the closest books.
- The distances are combined as:

```math
distance = \alpha \cdot distance_{title} + (1 - \alpha) \cdot distance_{genre}
```

- Books are ranked based on **distance** (lower is better).

---

## The Curiosity Gauge

The **curiosity gauge** lets users explore books **beyond the top-ranked recommendations**.

| Curiosity Level | Recommendation Range |
|-----------------|---------------------|
| `1` (Default)  | **Top-ranked books** |
| `2`            | **Starting from 0.01% of the dataset** |
| `3`            | **Starting from 0.03% of the dataset** |
| `4`            | **Starting from 0.06% of the dataset** |

> Example: If there are **100,000 books**, curiosity `3` starts from book **3,000+** instead of the top-ranked books.

### **How it works:**
- The recommendation list is **sorted by similarity/distance**.
- Instead of always picking the **top N books**, the system starts **at an offset** based on `curiosity`.

---

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cosine_similarity` | Use cosine similarity (`True`) or KNN (`False`) | `True` |
| `n_neighbors` | Number of books to recommend | `3` |
| `alpha` | Weight of **title/description** vs. **genre** (0 = only genre, 1 = only title) | `0.5` |
| `curiosity` | Explore deeper recommendations (`1-4`) | `1` |
