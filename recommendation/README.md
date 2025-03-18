# 📚 CamemBERT-KNN Book Recommendation System

This repository implements a **book recommendation system** using **CamemBERT embeddings** and a **hybrid search approach** (cosine similarity or k-Nearest Neighbors). The system incorporates **genre embeddings** and a **curiosity gauge** to fine-tune recommendations.

## Features
- **Uses embeddings** for book titles and descriptions combined, and genres.
- **Supports two recommendation methods**:
  - **Cosine Similarity**: Measures the similarity between books.
  - **k-Nearest Neighbors (KNN)**: Finds the most similar books in the dataset.
- **Robust Scaling**: Ensures balanced contribution from title/description and genre embeddings.
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

#### **A) Cosine Similarity (with Robust Scaling)**
- We compute **separate cosine similarities**:

```math
similarity_{titledesc} = \cos(dataset\_embedding_{titledesc}, input\_embedding_{titledesc})
```

```math
similarity_{genre} = \cos(dataset\_embedding_{genre}, input\_embedding_{genre})
```

- To ensure balanced influence from each embedding type, we apply **Min-Max Scaling** separately to both sets of similarities, bringing them into a consistent `[0,1]` range:

```math
scaled\_similarity = \frac{similarity - similarity_{min}}{similarity_{max} - similarity_{min}}
```

- The final similarity score is a **weighted combination**:

```math
similarity = \alpha \cdot scaled\_similarity_{titledesc} + (1 - \alpha) \cdot scaled\_similarity_{genre}
```

- Books are ranked based on **1 - similarity** (lower is better).

#### **B) k-Nearest Neighbors (KNN)**
- We train separate **KNN models** for **title/description** and **genre**.
- Each model returns the closest books.
- The distances are combined as:

```math
distance = \alpha \cdot distance_{titledesc} + (1 - \alpha) \cdot distance_{genre}
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
| `n_neighbors` | Number of books to recommend | `3` |
| `alpha` | Weight of **title/description** vs. **genre** (0 = genre more important, 1 = genre not important) | `0.1` |
| `curiosity` | Explore deeper recommendations (`1-4`) | `1` |
