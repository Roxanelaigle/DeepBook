import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict

from covers_to_text.main import picture_reader
from gbooks_api.main import gbooks_lookup
from recommendation.main import main as recommender

def main_pipeline(input_book: np.array,
                  curiosity: int,
                  embeddings_file_path: str,
                  dataset_path: str) -> Dict:
    # 1st step: Read the image and extract text
    extracted_words: str = picture_reader(input_book)
    # 2nd step: Look up the book on Google Books
    input_book: Dict = gbooks_lookup(extracted_words)
    # 3rd step: Recommend books based on the input book
    recommended_books: Dict = recommender(input_book,
                                          dataset_path,  # Path to the dataset
                                          embeddings_file_path,  # Path to the embeddings file
                                          curiosity)
    return recommended_books

if __name__ == "__main__":
    image_path = Path("raw_data/image.jpg")
    image = Image.open(image_path)
    input_book = np.array(image)
    curiosity = 2
    dataset_path = Path("raw_data/VF_data_base_consolidate_clean.csv")
    n_books = pd.read_csv(dataset_path).shape[0]
    embeddings_file_path = Path(f"models/camembert_models/embeddings_camemBERT_{n_books}_books.npy")
    recommended_books = main_pipeline(input_book, curiosity, embeddings_file_path, dataset_path)
    print(recommended_books)
