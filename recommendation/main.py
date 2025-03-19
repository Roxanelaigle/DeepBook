import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Dict, List, Union

from recommendation.save_load import load_embeddings, save_embeddings
from recommendation.embeddings import get_embeddings, get_input_embedding
from recommendation.preprocessing import load_dataset, prepare_text_features
from recommendation.recommender import recommend_books
import json

def main(input_book: Union[Dict, List[Dict]],
         dataset_path: str,
         model_dir: str,
         curiosity: int = 1,
         n_neighbors: int = 1,
         embeddings_sources: List[str] = ["titledesc"],
         alpha: float = 0.5) -> Dict:
    """
    Main script to load dataset, generate/load embeddings, fit KNN model, and recommend books.

    - Loads and prepares the dataset.
    - Loads or generates embeddings.
    - Fits a KNN model and recommends books.
    - Alpha is the weight of the genre embeddings.
    """

    if (len(embeddings_sources) != 1 and embeddings_sources == ["titledesc"])\
        or (len(embeddings_sources) > 2 and embeddings_sources == ["titledesc", "genre"]):
        raise ValueError('embeddings_sources should be 1 or 2 elements: ["titledesc"] or ["titledesc", "genre"]')

    logger.info("Loading dataset...")
    df = load_dataset(dataset_path)
    n_books = pd.read_csv(dataset_path).shape[0]
    logger.success(f"Dataset loaded successfully. Total records: {n_books}")

    logger.info("Preparing text features...")
    df = prepare_text_features(df)
    logger.success("Text features prepared successfully.")

    embeddings_dict = {}
    for source in embeddings_sources:
        embeddings_file_path = Path(model_dir) / f"embeddings_camemBERT_{source}_{n_books}_books.npy"
        try:
            logger.info(f"🔎 Attempting to load precomputed embeddings from {embeddings_file_path}...")
            embeddings_dict[source] = load_embeddings(model_dir, source, n_books)
            logger.success(f"✅ Precomputed embeddings for {source} loaded successfully.")
        except FileNotFoundError:
            logger.warning(f"❌ Precomputed embeddings not found at {embeddings_file_path}.")
            logger.info(f"⏳ Generating new embeddings for {source}...")
            embeddings = get_embeddings(df['combined_features'].tolist())
            embeddings_dict[source] = embeddings
            save_embeddings(embeddings, model_dir, source, n_books)
            logger.success(f"💾 New embeddings for {source} generated and saved.")

    df['embeddings'] = list(embeddings_dict["titledesc"])

    genre = "genre" in embeddings_sources
    if genre:
        df['embeddings_genre'] = list(embeddings_dict["genre"])

    # Get or generate embeddings for the input book or books
    if isinstance(input_book, dict):
        input_embedding = get_input_embedding(input_book, df)
    elif isinstance(input_book, list): # If multiple books, generate barycenter
        # First, remove None values from the list and log how many were removed
        len_before = len(input_book)
        input_book = [book for book in input_book if book is not None]
        logger.warning(f"⚠️ {len_before - len(input_book)} books were not found.")
        logger.info(f"Computing recommendations with the {len(input_book)} remaining books.")
        input_texts = [book['Title'] + " " + book['Description'] for book in input_book]
        logger.info(f"Generating embeddings for {len(input_book)} input books: {', '.join([book['Title'] for book in input_book])}")
        input_embeddings = [get_input_embedding(book, df) for book in input_book]
        # We get the mean of the embeddings for the multiple books
        input_embedding = sum(input_embeddings) / len(input_embeddings)
    else:
        raise ValueError("Input must be a dictionary or a list of dictionaries")

    if genre:
        if isinstance(input_book, dict):
            input_genre_embedding = get_input_embedding(input_book, df, embedding_type="genre")
        else: # If multiple books, generate barycenter
            input_genre_embeddings = [get_input_embedding(book, df, embedding_type="genre") for book in input_book]
            input_genre_embedding = sum(input_genre_embeddings) / len(input_genre_embeddings)
        assert input_embedding.shape == input_genre_embedding.shape, "Embedding dimensions do not match!"
        logger.success("✅ Input book embeddings generated for genre and title/description.")
    else:
        input_genre_embedding = None  # No genre embedding if not available
        logger.warning("⚠️ Input book embeddings generated for title/description only.")

    # Call `recommend_books` using separate embeddings
    logger.info("Generating recommendations...")
    recommended_books = recommend_books(
        df,
        input_embedding,
        curiosity=curiosity,
        n_neighbors=n_neighbors,
        genre_embedding=input_genre_embedding,  # Passing genre embedding separately
        alpha=alpha
    )

    logger.success("✅ Recommendations generated successfully.")
    logger.info(f"Recommended Books:\n{recommended_books}")
    # first fill result with the input book or books
    if isinstance(input_book, dict):
        input_book_json = {
                "title": input_book['Title'],
                "authors": input_book['Authors'],
                "image_link": input_book['Image Link'],
                "isbn": input_book['ISBN-13'],
                "description": input_book['Description']
            }
    else:
        input_book_json = [
            {
                "title": book['Title'],
                "authors": book['Authors'],
                "image_link": book['Image Link'],
                "isbn": book['ISBN-13'],
                "description": book['Description']
            } for book in input_book
        ]

    result = {
        "input_book": input_book_json,
        "output_books": [
            {
                "title": book['Title'],
                "authors": book['Authors'],
                "image_link": book['Image Link'],
                "isbn": book['ISBN-13'],
                "description": book['Description']
            } for _, book in recommended_books.iterrows()
        ]
    }

    return result



if __name__ == "__main__":
    embeddings_sources = ["titledesc", "genre"] # ["titledesc"] or ["titledesc", "genre"]
    curiosity = 1
    n_neighbors = 3
    alpha = 0.2 # Adjust this to control genre influence
    dataset_path = Path("raw_data/clean_data.csv")
    model_dir = Path(f"models/camembert_models/")

    n_books = pd.read_csv(dataset_path).shape[0]
    logger.info(f"Dataset size determined: {n_books} books.")

    input_book = {
        'Title': "Sam et Cléo, c'est le monde à l'envers - Qu'est-ce qu'on dit, les parents ?",
        'Authors': 'Héloïse Junier, Arthur Du Coteau',
        'Publisher': 'Hatier Jeunesse',
        'Published Date': '2025-04-09',
        'Categories': 'Juvenile Fiction',
        'Description': "Aujourd’hui, Sam et Cléo sont invités à dîner chez des amis...",
        'Page Count': 0,
        'Language': 'fr',
        'ISBN-10': '2401116451',
        'ISBN-13': '9782401116450',
        'Preview Link': 'http://books.google.fr/books?id=6IFIEQAAQBAJ',
        'Info Link': 'http://books.google.fr/books?id=6IFIEQAAQBAJ',
        'Average Rating': 'Non noté',
        'Ratings Count': 0,
        'Image Link': 'http://books.google.com/books/content?id=6IFIEQAAQBAJ',
        'Saleability': 'NOT_FOR_SALE',
        'Price': 'Non disponible',
        'Currency': 'Non disponible'
    }

    recommended_books = main(input_book,
                             dataset_path,
                             model_dir,
                             curiosity,
                             n_neighbors,
                             embeddings_sources,
                             alpha=alpha)

    print()
    print("Recommended Book:")
    print(json.dumps(recommended_books, indent=4, ensure_ascii=False))
