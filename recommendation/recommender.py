import numpy as np
import pandas as pd
from loguru import logger
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim

def fit_knn(embeddings: np.ndarray, n_neighbors: int = 10) -> NearestNeighbors:
    """"
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
                    alpha: float = 0.5) -> pd.DataFrame:
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
    genre = genre_embedding is not None  # Check if genre embeddings are provided

    if cosine_similarity:
        # Ensure embeddings are available
        if 'embeddings' not in database.columns:
            raise KeyError("Missing 'embeddings' in database.")
        if genre and 'embeddings_genre' not in database.columns:
            raise KeyError("Missing 'embeddings_genre' in database.")

        # Compute cosine similarities separately
        similarities_titledesc = cosine_sim(
            np.vstack(database['embeddings'].values),
            input_embedding.reshape(1, -1)
        )

        similarities_genre = (cosine_sim(
            np.vstack(database['embeddings_genre'].values),
            genre_embedding.reshape(1, -1)
        ) if genre else np.zeros_like(similarities_titledesc))

        # Combine similarities with weighting factor alpha
        similarities = alpha * similarities_titledesc + (1 - alpha) * similarities_genre

        # Convert similarities into distances (1 - similarity) and sort
        distances = 1 - similarities
        sorted_indices = np.argsort(distances.flatten())

    else:
        # Use KNN-based approach instead of cosine similarity
        logger.info("Fitting KNN models separately for each embedding type...")

        knn_titledesc = fit_knn(np.vstack(database['embeddings'].values), n_neighbors=len(database))
        distances_titledesc, indices_titledesc = knn_titledesc.kneighbors(input_embedding.reshape(1, -1), n_neighbors=len(database))

        if genre:
            knn_genre = fit_knn(np.vstack(database['embeddings_genre'].values), n_neighbors=len(database))
            distances_genre, indices_genre = knn_genre.kneighbors(genre_embedding.reshape(1, -1), n_neighbors=len(database))

            # Merge distances with weight alpha
            combined_scores = alpha * distances_titledesc + (1 - alpha) * distances_genre
            sorted_indices = np.argsort(combined_scores.flatten())
        else:
            sorted_indices = indices_titledesc.flatten()

    total_books = len(database)

    # **Curiosity handling: defines the starting position in the ranked results**
    if curiosity == 1:  # Top results (starting from rank 0)
        start_index = 0
    elif curiosity == 2:  # Start at 0.01% of dataset
        start_index = min(int(0.01 * total_books), total_books - 1)
    elif curiosity == 3:  # Start at 0.03% of dataset
        start_index = min(int(0.03 * total_books), total_books - 1)
    elif curiosity == 4:  # Start at 0.06% of dataset
        start_index = min(int(0.06 * total_books), total_books - 1)
    else:
        raise ValueError("Curiosity must be 1, 2, 3, or 4.")

    end_index = min(start_index + n_neighbors, total_books)

    # Select book indices based on curiosity range
    recommended_indices = sorted_indices[start_index:end_index]

    recommended_books = database.iloc[recommended_indices]

    return recommended_books[['Title', 'Authors', 'Categories', 'Description', 'ISBN-13', 'Image Link']]
