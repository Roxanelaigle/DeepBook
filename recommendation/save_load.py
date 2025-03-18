from io import BytesIO

import numpy as np
from pathlib import Path
from typing import Tuple
from transformers import CamembertTokenizer, TFCamembertModel
from google.cloud import storage

def save_embeddings(embeddings: np.ndarray, model_dir: Path, text_type: str, n_embeddings: int):
    """Save embeddings to a file."""
    filename = model_dir / f"embeddings_camemBERT_{text_type}_{n_embeddings}_books.npy"  # Construct the full file path
    filename.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    np.save(filename, embeddings)

def load_embeddings(model_dir: Path, text_type: str, n_embeddings: int) -> np.ndarray:
    """Load embeddings from a file."""
    filename = model_dir / f"embeddings_camemBERT_{text_type}_{n_embeddings}_books.npy"  # Construct the full file path

    # Check if the file exists in the bucket
    # client = storage.Client()
    # bucket = client.bucket("le_seau-deepbook")
    # blob = bucket.blob(str(filename))

    # if blob.exists():
    #     print(f"Loading embeddings from {filename} in the bucket.")
    #     return np.load(BytesIO(blob.download_as_bytes()))
    # Return the embeddings if the file exists in the bucket else check if the file exists locally

    if filename.exists():
        return np.load(filename)
    else:
        raise FileNotFoundError(f"{filename} not found.")

def save_model(model: TFCamembertModel, tokenizer: CamembertTokenizer, path: Path):
    """Save model and tokenizer to a file."""
    path = Path(path)
    tokenizer.save_pretrained(str(path))
    model.save_pretrained(str(path))

def load_model(path: Path) -> Tuple[TFCamembertModel, CamembertTokenizer]:
    """Load model and tokenizer from a file."""
    path = Path(path)
    tokenizer = CamembertTokenizer.from_pretrained(path)
    model = TFCamembertModel.from_pretrained(path)
    return model, tokenizer
