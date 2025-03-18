import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List, Union, Literal

from covers_to_text.main import picture_reader, picture_reader_multibooks
from gbooks_api.main import gbooks_lookup, gbooks_lookup_multibooks, gbooks_look_isbn
from recommendation.main import main as recommender

def main_pipeline(input_book: Union[np.array, str],
                  photo_type: Literal["single", "multiple"],
                  curiosity: int,
                  model_dir: str,
                  dataset_path: str,
                  cosine_similarity: bool = False,
                  n_neighbors: int = 1,
                  embeddings_sources: List[str] = ["titledesc"],
                  alpha: float = 0.5) -> Dict:
    """
    Main pipeline to recommend books based on an input book.

    Args:
		input_book: Either an image vector or an ISBN (10 or 13)
		curiosity: Exploration level for recommendations (1-4)
		model_dir: Path to the model directory
		dataset_path: Path to the dataset
		cosine_similarity: Use cosine similarity or KNN distance
		n_neighbors: Number of similar books to recommend
		embeddings_sources: List of embeddings sources, either ["titledesc"] or ["titledesc", "genre"]
		alpha: Weight factor for title/description vs. genre similarity (0 = only genre, 1 = only title/description)

    Returns:
        A dictionary containing the recommended books.
    """

    # Identify input type (image or ISBN)
    if isinstance(input_book, np.ndarray):
        input_type = "image_vector"
    elif isinstance(input_book, str):
        if not (input_book.isdigit() and (len(input_book) == 10 or len(input_book) == 13)):
            raise ValueError("The input string is not a valid ISBN (10 or 13 digits)")
        input_type = "ISBN"
    else:
        raise ValueError("Input must be either an image vector (np.array) or an ISBN string")

    # Step 1: Extract text from image (if applicable)
    if input_type == "image_vector":
        if photo_type == "single":
            extracted_words: List[str] = picture_reader(input_book)
            # Step 2 (single book): Look up book details on Google Books
            input_book: Dict = gbooks_lookup(extracted_words)
        else:
            # List of extracted words from multiple books
            list_extracted_words: List[List[str]] = picture_reader_multibooks(input_book)
            # Step 2 (multiple books): Look up book details on Google Books
            input_book: List[Dict] = gbooks_lookup_multibooks(list_extracted_words)
    else: # if ISBN
        # Step 2: ISBN - Look up book details on Google Books
        input_book = gbooks_look_isbn(input_book)

    # Step 3: Generate recommendations based on the input book
    recommended_books: Dict = recommender(
        input_book, # If multiple books, generate barycenter
        dataset_path,  # Path to the dataset
        model_dir,  # Path to the embeddings
        curiosity,
        n_neighbors=n_neighbors,
        cosine_similarity=cosine_similarity,
        embeddings_sources=embeddings_sources,
        alpha=alpha  # Weight for genre/title balance
    )
    return recommended_books


if __name__ == "__main__":
    embeddings_sources = ["titledesc", "genre"]  # Available options: ["titledesc"], ["titledesc", "genre"]
    # Curiosity level for recommendation
    curiosity = 1
    # Use cosine similarity (True) or KNN distance (False)
    cosine_similarity = True
    # Number of neighbors (books) to recommend
    n_neighbors = 3
    # Weighting factor for title/description vs. genre embeddings
    alpha = 1  # Adjust this to control genre influence

    # Dataset path
    dataset_path = Path("raw_data/VF_data_base_consolidate_clean.csv")
    n_books = pd.read_csv(dataset_path).shape[0]
    model_dir = Path(f"models/camembert_models/")

    # # Example: Use an ISBN (10 or 13) as input
    # input_book = "9782070643066"

    # Alternative: Use an image input (uncomment to test)
    image_path = Path("raw_data/cover.jpg")
    image = Image.open(image_path)
    input_book = np.array(image)

    # Run the recommendation pipeline
    recommended_books = main_pipeline(
        input_book,
        curiosity,
        model_dir,
        dataset_path,
        cosine_similarity,
        n_neighbors=n_neighbors,
        embeddings_sources=embeddings_sources,
        alpha=alpha
    )

    # Display results
    print()
    print(f"=== Input: {recommended_books['input_book']['title']} by {recommended_books['input_book']['authors']} ===")
    print(f"=== Input ISBN: {recommended_books['input_book']['isbn']} ===")
    print(f"=== {'Cosine Similarity' if cosine_similarity else 'KNN Distance'} ===")
    print(f"=== Curiosity: {curiosity} ===")
    print(f"=== Neighbors: {n_neighbors} ===")
    print(":")
    for reco in recommended_books["output_books"]:
        print(f"Recommended Book: {reco['title']} by {reco['authors']}")
        print(f"ISBN: {reco['isbn']}")
        print(f"Description: {reco['description']}")
        print("---")
