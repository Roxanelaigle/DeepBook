import pandas as pd
from pathlib import Path
from typing import List

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    """
    return pd.read_csv(Path(path))

def prepare_text_features(database: pd.DataFrame,
                          features: List[str] = ['Title', 'Description']) -> pd.DataFrame:
    """
    Combines the specified text feature columns into a single 'combined_features' column.
    Default features are 'Title' and 'Description'.
    """
    database['combined_features'] = database[features[0]] + " " + database[features[1]]
    return database
