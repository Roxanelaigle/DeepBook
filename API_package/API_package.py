
from fastapi import FastAPI, Request
import json
import numpy as np
from main_pipeline.main import main_pipeline
import pandas as pd
from pathlib import Path

app = FastAPI()
app.state.dataset_path = Path("data/VF_data_base_consolidate_clean.csv")
app.state.n_books = pd.read_csv(app.state.dataset_path).shape[0]
app.state.embeddings_file_path = Path(f"models/camembert_models/embeddings_camemBERT_{app.state.n_books}_books.npy")

@app.post("/")
async def predict(request: Request):

    raw_data = await request.body()
    data=json.loads(raw_data)
    curiosity_level = int(data['curiosity_level'])
    print("curiosity_level : " + str(curiosity_level))
    if 'image_array' in data.keys() :
        image_array = np.array(data['image_array'], dtype=np.uint8)
        return main_pipeline(image_array, curiosity_level,app.state.embeddings_file_path, app.state.dataset_path)
    else :
        isbn = data['isbn']
        print(isbn)
        return main_pipeline(isbn, curiosity_level, app.state.embeddings_file_path, app.state.dataset_path)
