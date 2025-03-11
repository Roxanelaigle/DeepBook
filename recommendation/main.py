from pathlib import Path
from typing import Dict
from save_load import load_embeddings, save_embeddings

from embeddings import get_embeddings
from preprocessing import load_dataset, prepare_text_features
from recommender import fit_knn, recommend_books
import pandas as pd


def main(input_book: Dict, dataset_path: str, embeddings_path: str, curiosity: int = 1):
    """
    Main script to load dataset, generate/load embeddings, fit KNN model, and recommend books.

    - The function first loads and prepares the dataset.
    - It then attempts to load precomputed embeddings; if not found, it generates and saves new embeddings.
    - The embeddings are added to the dataset.
    - A KNN model is fitted using the embeddings.
    - A sample book is used to generate an input embedding for recommendation.
    - The function prints a list of recommended books based on the input book's embedding and genre.
    """
    # Load the dataset and prepare text features (combine features into a single column)
    database = load_dataset(dataset_path)
    database = prepare_text_features(database)

    try: # Try to load precomputed embeddings
        embeddings = load_embeddings(embeddings_path)
    except FileNotFoundError: # If not found, generate and save new embeddings
        embeddings = get_embeddings(database['combined_features'].tolist())
        save_embeddings(embeddings, embeddings_path, len(embeddings))

    # Add embeddings to the database so we can use them for recommendations
    database['embeddings'] = list(embeddings)

    knn_model = fit_knn(embeddings) # Fit KNN model to the embeddings

    input_text = input_book['Title'] + " " + input_book['Description']
    input_embedding = get_embeddings([input_text])[0] # Generate embedding for the input book

    # Recommend books based on the input embedding and genre
    recommended_books = recommend_books(database,
                                        input_embedding,
                                        knn_model,
                                        book_genre=input_book['Genre'],
                                        curiosity=curiosity)
    print(recommended_books)

if __name__ == "__main__":
    # Test the main function over a sample book on curiosity level 1
    curiosity = 1
    dataset_path = Path("raw_data/google_books_consolidate_final.csv")

    # Get the size of the dataset
    n_books = pd.read_csv(dataset_path).shape[0]

    # Update the path to include the number of books
    embeddings_path = Path(f"models/camembert_model/embeddings_camemBERT_{str(n_books)}_books.npy")
    input_book = {'Title': 'Sample Book', 'Description': 'A thrilling adventure.', 'Genre': 'Fiction'}

    main(input_book, dataset_path, embeddings_path, curiosity)
