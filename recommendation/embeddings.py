import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from typing import Dict, List, Literal
from transformers import CamembertTokenizer, TFCamembertModel

tokenizer = CamembertTokenizer.from_pretrained('camembert-base', local_files_only=True)
model = TFCamembertModel.from_pretrained('camembert-base', local_files_only=True)

def get_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts using a pre-trained model.

    - The function processes the texts in batches to efficiently handle large datasets and avoid memory issues.
    - The embeddings are generated using a pre-trained model (CamemBERT)
    - The mean of the last hidden state is used as the embedding for each text.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[i:i+batch_size], return_tensors='tf',
            truncation=True, padding=True, max_length=512
        )
        # Process the batch on GPU
        with tf.device('/GPU:0'):
            output = model(batch)
            batch_emb = tf.reduce_mean(output.last_hidden_state, axis=1).numpy()
        embeddings.append(batch_emb)
    return np.vstack(embeddings)


def get_input_embedding(input_book: Dict,
                        df: pd.DataFrame,
                        embedding_type: Literal["titledesc", "genre"] = "titledesc") -> np.ndarray:
    """
    Get the embedding for the input book. If the book is in the dataset, use the existing embedding.
    Otherwise, generate a new embedding.
    """
    if df["ISBN-13"].str.contains(input_book["ISBN-13"]).any():
        logger.info(f'Input book found in the dataset: {input_book["Title"]}')
        input_embedding = df.loc[df["ISBN-13"] == input_book["ISBN-13"], "embeddings"].values[0] if embedding_type == "titledesc" else df.loc[df["ISBN-13"] == input_book["ISBN-13"], "embeddings_genre"].values[0]
    else:
        input_text = input_book["Title"] + " " + input_book["Description"]
        logger.info(f'Generating embeddings for input book: {input_book["Title"]}')
        input_embedding = get_embeddings([input_text])[0] if embedding_type == "titledesc" else get_embeddings([input_text])[0]
    return input_embedding
