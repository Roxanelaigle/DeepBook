from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

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
                    book_genre: str,
                    curiosity: int = 1,
                    n_neighbors: int = 5) -> pd.DataFrame:
    """
    Recommend books based on the input embedding and genre.
    Output the top n_neighbors recommendations as a DataFrame.
    """
    if curiosity == 1:
         # filter by genre
        candidates = database[database['Genre'] == book_genre]
    elif curiosity == 2:
        # do not filter by genre
        candidates = database
    elif curiosity == 3:
        # filter by other genres
        candidates = database[database['Genre'] != book_genre]
    else:
        raise ValueError("Invalid curiosity level. Choose 1, 2, or 3.")

    if candidates.empty:
        return pd.DataFrame()

    candidate_embeddings = np.vstack(candidates['embeddings'].values)
    distances, indices = knn_model.kneighbors([input_embedding],
                                              n_neighbors=min(n_neighbors+1, len(candidate_embeddings)))

    if curiosity == 3:
        recommended = candidates.iloc[indices[0][-n_neighbors:]]
    else:
        recommended = candidates.iloc[indices[0][1:n_neighbors+1]]

    return recommended[['Title', 'Genre']]
