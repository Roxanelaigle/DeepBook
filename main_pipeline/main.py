import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, Union

from covers_to_text.main import picture_reader
from gbooks_api.main import gbooks_lookup, gbooks_look_isbn
from recommendation.main import main as recommender

def main_pipeline(input_book: Union[np.array, str],
                  curiosity: int,
                  embeddings_file_path: str,
                  dataset_path: str) -> Dict:
    """
    Main pipeline to recommend books based on an input book.
    Args:
        input_book: Either an image vector or an ISBN (10 or 13)
        curiosity: Number of books to recommend
        embeddings_file_path: Path to the embeddings file
        dataset_path: Path to the dataset

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
                                          embeddings_file_path,  # Path to the embeddings file
                                          curiosity)
    return recommended_books

if __name__ == "__main__":
    # # Test the main pipeline with an image
    image_path = Path("raw_data/image.jpg")
    image = Image.open(image_path)
    input_book = np.array(image)

    # # Test the main pipeline with an ISBN 13
    # input_book = "9782807906266"

    # Test the main pipeline with an ISBN 10
    # input_book = "2807906265"

    curiosity = 2
    dataset_path = Path("raw_data/VF_data_base_consolidate_clean.csv")
    n_books = pd.read_csv(dataset_path).shape[0]
    embeddings_file_path = Path(f"models/camembert_models/embeddings_camemBERT_{n_books}_books.npy")
    recommended_books = main_pipeline(input_book, curiosity, embeddings_file_path, dataset_path)
