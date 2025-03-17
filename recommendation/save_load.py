import numpy as np
from pathlib import Path
from typing import Tuple
from transformers import CamembertTokenizer, TFCamembertModel

def save_embeddings(embeddings: np.ndarray, model_dir: Path, text_type: str, n_embeddings: int):
    """Save embeddings to a file."""
    filename = model_dir / f"embeddings_camemBERT_{text_type}_{n_embeddings}_books.npy"  # Construct the full file path
    filename.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    np.save(filename, embeddings)

def load_embeddings(model_dir: Path, text_type: str, n_embeddings: int) -> np.ndarray:
    """Load embeddings from a file."""
    filename = model_dir / f"embeddings_camemBERT_{text_type}_{n_embeddings}_books.npy"  # Construct the full file path
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
