import numpy as np
import tensorflow as tf
from typing import List
from transformers import CamembertTokenizer, TFCamembertModel

tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = TFCamembertModel.from_pretrained('camembert-base')

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
