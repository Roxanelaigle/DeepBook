import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Dict, List

from recommendation.save_load import load_embeddings, save_embeddings
from recommendation.embeddings import get_embeddings
from recommendation.preprocessing import load_dataset, prepare_text_features
from recommendation.recommender import fit_knn, recommend_books
import json


def main(input_book: Dict,
         dataset_path: str,
         model_dir: str,
         curiosity: int = 1,
         n_neighbors: int = 1,
         cosine_similarity: bool = False,
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

    # Generate embeddings for the input book
    input_text = input_book['Title'] + " " + input_book['Description']
    logger.info(f"Generating embeddings for input book: {input_book['Title']}")
    input_embedding = get_embeddings([input_text])[0]

    if genre:
        input_genre_embedding = get_embeddings([input_book['Categories']])[0]
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
        book_genre=input_book['Categories'],
        curiosity=curiosity,
        n_neighbors=n_neighbors,
        cosine_similarity=cosine_similarity,
        genre_embedding=input_genre_embedding,  # Passing genre embedding separately
        alpha=alpha
    )

    logger.success("✅ Recommendations generated successfully.")
    logger.info(f"Recommended Books:\n{recommended_books}")

    result = {
        "input_book": {
            "title": input_book['Title'],
            "authors": input_book['Authors'],
            "image_link": input_book['Image Link'],
            "isbn": input_book['ISBN-13'],
            "description": input_book['Description']
        },
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
    cosine_similarity = True
    alpha = 0.2 # Adjust this to control genre influence
    dataset_path = Path("raw_data/VF_data_base_consolidate_clean.csv")
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
                             cosine_similarity,
                             embeddings_sources,
                             alpha=alpha)

    print()
    print("Recommended Book:")
    print(json.dumps(recommended_books, indent=4, ensure_ascii=False))
