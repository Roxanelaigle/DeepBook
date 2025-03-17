import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List, Union

from covers_to_text.main import picture_reader
from gbooks_api.main import gbooks_lookup, gbooks_look_isbn
from recommendation.main import main as recommender

def main_pipeline(input_book: Union[np.array, str],
                  curiosity: int,
                  model_dir: str,
                  dataset_path: str,
                  cosine_similarity: bool = False,
                  n_neighbors: int = 1,
                  embeddings_sources: List[str] = ["titledesc"]) -> Dict:
    """
    Main pipeline to recommend books based on an input book.
    Args:
        input_book: Either an image vector or an ISBN (10 or 13)
        curiosity: Number of books to recommend
        model_dir: Path to the model directory
        dataset_path: Path to the dataset
        cosine_similarity: Flag to use cosine similarity or KNN distance
        n_neighbors: Number of neighbors to consider in the KNN algorithm
        embeddings_sources: List of sources for the embeddings, either ["titledesc"] or ["titledesc", "genre"]

    Returns:
        recommended_books: A dictionary containing the recommended books information
    """
    if isinstance(input_book, np.ndarray):
        input_type = "image_vector"
    elif isinstance(input_book, str):
        if not (input_book.isdigit() and (len(input_book) == 10 or len(input_book) == 13)):
            raise ValueError("The input string is not an ISBN (10 or 13)")
        input_type = "ISBN"
    else:
        raise ValueError("The input should be an image vector or an ISBN (10 or 13)")
    # 1st step: If image, extract its text content
    extracted_words: str = picture_reader(input_book) if input_type == "image_vector" else input_book
    # 2nd step: Look up the book on Google Books
    input_book: Dict = gbooks_lookup(extracted_words) if input_type == "image_vector" else gbooks_look_isbn(input_book)
    # 3rd step: Recommend books based on the input book
    recommended_books: Dict = recommender(input_book,
                                          dataset_path,  # Path to the dataset
                                          model_dir,  # Path to the embeddings file
                                          curiosity,
                                          cosine_similarity=cosine_similarity,
                                          n_neighbors=n_neighbors,
                                          embeddings_sources=embeddings_sources)
    return recommended_books

if __name__ == "__main__":
    embeddings_sources = ["titledesc"] # ["titledesc"] or ["titledesc", "genre"]
    # Set the curiosity level
    curiosity = 4
    # Set the cosine similarity flag - cosine_similarity or KNN distance
    cosine_similarity = True
    # Set the number of neighbors to consider
    n_neighbors = 3

    # Test the main pipeline with an image
    # image_path = Path("raw_data/cover.jpg")
    # image = Image.open(image_path)
    # input_book = np.array(image)

    # Test the main pipeline with an ISBN 13
    input_book = "9782070643066"

    # # Test the main pipeline with an ISBN 10
    # input_book = "2807906265"
    dataset_path = Path("raw_data/VF_data_base_consolidate_clean.csv")
    n_books = pd.read_csv(dataset_path).shape[0]
    model_dir = Path(f"models/camembert_models/")
    recommended_books = main_pipeline(input_book,
                                      curiosity,
                                      model_dir,
                                      dataset_path,
                                      cosine_similarity,
                                      n_neighbors=n_neighbors,
                                      embeddings_sources=embeddings_sources)
    print()
    print(f"=== Input: {recommended_books['input_book']['title']} by {recommended_books['input_book']['authors']} ===")
    print(f"=== Input ISBN: {recommended_books['input_book']['isbn']} ===")
    print(f"=== {'Cosine Similarity' if cosine_similarity else 'KNN Distance'} ===")
    print(f"=== Curiosity: {curiosity} ===")
    print(f"=== Neighbors: {n_neighbors} ===")
    print(":")
    for reco in recommended_books["output_books"]:
        print(f"Recommended Book: \n{reco['title']} by {reco['authors']}")
        print()
        print(f"ISBN: {reco['isbn']}")
        print()
        print(f"Description: \n{reco['description']}")
        print("---")
        print()
