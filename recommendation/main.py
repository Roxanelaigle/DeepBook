import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Dict

from recommendation.save_load import load_embeddings, save_embeddings
from recommendation.embeddings import get_embeddings
from recommendation.preprocessing import load_dataset, prepare_text_features
from recommendation.recommender import fit_knn, recommend_books
import json


def main(input_book: Dict,
         dataset_path: str,
         embeddings_file_path: str,
         curiosity: int = 1,
         n_neighbors: int = 1) -> Dict:
    """
    Main script to load dataset, generate/load embeddings, fit KNN model, and recommend books.

    - Loads and prepares the dataset.
    - Loads or generates embeddings.
    - Fits a KNN model and recommends books.
    """
    logger.info("Loading dataset...")
    df = load_dataset(dataset_path)
    logger.success(f"Dataset loaded successfully. Total records: {len(df)}")

    logger.info("Preparing text features...")
    df = prepare_text_features(df)
    logger.success("Text features prepared successfully.")

    try:
        logger.info(f"🔎 Attempting to load precomputed embeddings from {embeddings_file_path}...")
        embeddings = load_embeddings(embeddings_file_path)
        logger.success("✅ Precomputed embeddings loaded successfully.")
    except FileNotFoundError:
        logger.warning(f"⏳ Precomputed embeddings not found at {embeddings_file_path}. Generating new embeddings...")
        embeddings = get_embeddings(df['combined_features'].tolist())
        save_embeddings(embeddings, Path(embeddings_file_path).parent, len(embeddings))
        logger.success("💾 New embeddings generated and saved.")

    df['embeddings'] = list(embeddings)

    logger.info("Fitting KNN model...")
    knn_model = fit_knn(embeddings)
    logger.success("KNN model fitted successfully.")

    input_text = input_book['Title'] + " " + input_book['Description']
    logger.info(f"Generating embedding for input book: {input_book['Title']}")
    input_embedding = get_embeddings([input_text])[0]
    logger.success("Input book embedding generated successfully.")

    logger.info("Generating recommendations...")
    recommended_books = recommend_books(
        df,
        input_embedding,
        knn_model,
        book_genre=input_book['Categories'],
        curiosity=curiosity,
        n_neighbors=n_neighbors
    )

    logger.success("Recommendations generated successfully.")
    logger.info(f"Recommended Book:\n{recommended_books}")
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
    curiosity = 2
    dataset_path = Path("raw_data/VF_data_base_consolidate_clean.csv")

    n_books = pd.read_csv(dataset_path).shape[0]
    logger.info(f"Dataset size determined: {n_books} books.")

    embeddings_file_path = Path(f"models/camembert_models/embeddings_camemBERT_{n_books}_books.npy")

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

    recommended_books = main(input_book, dataset_path, embeddings_file_path, curiosity)

    print()
    print("Recommended Book:")
    print(json.dumps(recommended_books, indent=4, ensure_ascii=False))
