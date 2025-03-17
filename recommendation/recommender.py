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
                    knn_model: NearestNeighbors,
                    book_genre: str = None,
                    curiosity: int = 1,
                    n_neighbors: int = 1,
                    cosine_similarity: bool = False) -> pd.DataFrame:
    """
    Recommend books based on the input embedding.
    Output the top n_neighbors recommendations as a DataFrame.
    """
    if cosine_similarity:
        # Compute cosine similarities
        if 'embeddings' not in database.columns:
            raise KeyError("The 'embeddings' column is not present in the database DataFrame.")

        similarities = cosine_sim(np.vstack(database['embeddings'].values), input_embedding.reshape(1, -1))
        distances = 1 - similarities  # Convert similarities to distances
        indices = np.argsort(distances.flatten())  # Sorted indices
    else:
        distances, indices = knn_model.kneighbors(input_embedding.reshape(1, -1), n_neighbors=len(database))

    total_books = len(database)

    # Compute the starting index based on curiosity
    if curiosity == 1:  # Top n_neighbors recommendations
        start_index = 0
    elif curiosity == 2:  # Recommendations starting from 0.01%
        start_index = min(int(0.01/100 * total_books), total_books - 1) # 3e voisin
    elif curiosity == 3:  # Recommendations starting from 0.03%
        start_index = min(int(0.03/100 * total_books), total_books - 1) #10e voisin
    elif curiosity == 4:  # Recommendations starting from 0.1%
        start_index = min(int(0.06/100 * total_books), total_books - 1) #20e voisin
    else:
        raise ValueError("Invalid curiosity level. Choose 1, 2, 3, or 4.")

    # Make sure we don't exceed the dataset size
    end_index = min(start_index + n_neighbors, total_books)

    logger.info(f"Start index: {start_index}, End index: {end_index}")
    logger.info(f"N Neighbors: {n_neighbors}, Total Books: {total_books}")

    # Fix slicing of indices
    recommended_indices = indices[start_index:end_index].tolist() if cosine_similarity else indices[0][start_index:end_index]

    logger.info(f"Start index: {start_index}; End index: {end_index}")  # Keeping your original log statement

    recommended_books = database.iloc[recommended_indices]

    return recommended_books[['Title', 'Authors', 'Categories', 'Description', 'ISBN-13', 'Image Link']]
