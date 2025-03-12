import numpy as np
from pathlib import Path
from typing import Tuple
from transformers import CamembertTokenizer, TFCamembertModel

def save_embeddings(embeddings: np.ndarray, path: Path, n_embeddings: int):
    """Save embeddings to a file."""
    filename = path
    path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    np.save(filename, embeddings)

def load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from a file."""
    path = Path(path)
    if path.exists():
        return np.load(path)
    else:
        raise FileNotFoundError(f"{path} not found.")

def save_model(model: TFCamembertModel, tokenizer: CamembertTokenizer, path: Path):
    """Save model and tokenizer to a file."""
    path = Path(path)
    tokenizer.save_pretrained(path)
    model.save_pretrained(path)

def load_model(path: Path) -> Tuple[TFCamembertModel, CamembertTokenizer]:
    """Load model and tokenizer from a file."""
    path = Path(path)
    tokenizer = CamembertTokenizer.from_pretrained(path)
    model = TFCamembertModel.from_pretrained(path)
    return model, tokenizer
