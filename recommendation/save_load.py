import numpy as np
from pathlib import Path
from typing import Tuple
from transformers import CamembertTokenizer, TFCamembertModel

def save_embeddings(embeddings: np.ndarray, path: Path, n_embeddings: int):
    """Save embeddings to a file."""
    filename = path / f"embeddings_camemBERT_{n_embeddings}_books.npy"
    np.save(filename, embeddings)

def load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from a file."""
    path = Path(path)
    if path.exists():
        return np.load(path)
    else:
        raise FileNotFoundError(f"{path} not found.")

def save_model(model: TFCamembertModel, tokenizer: CamembertTokenizer, save_directory: Path):
    """Save model and tokenizer to a directory."""
    save_directory = Path(save_directory)
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

def load_model(save_directory: Path) -> Tuple[TFCamembertModel, CamembertTokenizer]:
    """Load model and tokenizer from a directory."""
    save_directory = Path(save_directory)
    tokenizer = CamembertTokenizer.from_pretrained(save_directory)
    model = TFCamembertModel.from_pretrained(save_directory)
    return model, tokenizer
