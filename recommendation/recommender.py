import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
from sklearn.preprocessing import normalize, MinMaxScaler


def fit_knn(embeddings: np.ndarray, n_neighbors: int = 10) -> NearestNeighbors:
    """
    Fit a KNN model to the embeddings.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(embeddings)
    return knn


def recommend_books(database: pd.DataFrame,
                    input_embedding: np.ndarray,
                    knn_model: NearestNeighbors = None,
                    book_genre: str = None,
                    curiosity: int = 1,
                    n_neighbors: int = 1,
                    cosine_similarity: bool = False,
                    genre_embedding: np.ndarray = None,
                    alpha: float = 0.1) -> pd.DataFrame:
    """
    Recommend books based on title/description and genre embeddings.
    - Uses either cosine similarity or KNN for recommendations.
    - Alpha controls the weighting of title/description vs. genre similarities.
    - Curiosity defines how far in the ranked list we start picking recommendations.

    Args:
        database: DataFrame containing book information and embeddings.
        input_embedding: The embedding of the input book (title/description).
        knn_model: Pre-trained KNN model (if used).
        book_genre: The genre of the input book (optional).
        curiosity: Defines how "far" into the ranked list we start recommendations (1-4).
        n_neighbors: Number of recommendations to return.
        cosine_similarity: Whether to use cosine similarity instead of KNN.
        genre_embedding: The embedding for the genre information (if available).
        alpha: Weighting factor (0 = only genre, 1 = only title/description).

    Returns:
        DataFrame with recommended books.
    """
    genre = genre_embedding is not None

    # Normalize embeddings
    db_embeddings_norm = normalize(np.vstack(database['embeddings'].values))
    input_embedding_norm = normalize(input_embedding.reshape(1, -1))

    similarities_titledesc = cosine_sim(db_embeddings_norm, input_embedding_norm).flatten()

    if genre:
        db_genre_embeddings_norm = normalize(np.vstack(database['embeddings_genre'].values))
        genre_embedding_norm = normalize(genre_embedding.reshape(1, -1))
        similarities_genre = cosine_sim(db_genre_embeddings_norm, genre_embedding_norm).flatten()
    else:
        similarities_genre = np.zeros_like(similarities_titledesc)

    # Min-Max Scaling of similarities
    scaler = MinMaxScaler()
    similarities_titledesc_scaled = scaler.fit_transform(similarities_titledesc.reshape(-1, 1)).flatten()
    similarities_genre_scaled = scaler.fit_transform(similarities_genre.reshape(-1, 1)).flatten()

    # Combine scaled similarities with alpha weighting
    combined_similarities = alpha * similarities_titledesc_scaled + (1 - alpha) * similarities_genre_scaled

    # Convert similarities to distances (1 - similarity)
    distances = 1 - combined_similarities
    sorted_indices = np.argsort(distances)

    total_books = len(database)

    # Curiosity handling
    if curiosity == 1:
        start_index = 0
    elif curiosity == 2:
        start_index = min(int(0.01 * total_books), total_books - 1)
    elif curiosity == 3:
        start_index = min(int(0.03 * total_books), total_books - 1)
    elif curiosity == 4:
        start_index = min(int(0.06 * total_books), total_books - 1)
    else:
        raise ValueError("Curiosity must be 1, 2, 3, or 4.")

    end_index = min(start_index + n_neighbors, total_books)

    recommended_indices = sorted_indices[start_index:end_index]

    recommended_books = database.iloc[recommended_indices]

    return recommended_books[['Title', 'Authors', 'Categories', 'Description', 'ISBN-13', 'Image Link']]
