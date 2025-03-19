
from fastapi import FastAPI, Request, File, UploadFile
import json
import numpy as np
from main_pipeline.main import main_pipeline
import pandas as pd
from pathlib import Path
from PIL import Image
import io
from fastapi import Form

app = FastAPI()
app.state.alpha = 0.9
app.state.embeddings_sources = ["titledesc", "genre"]
app.state.dataset_path = Path("raw_data/VF_data_base_consolidate_clean.csv")
app.state.n_books = pd.read_csv(app.state.dataset_path).shape[0]
app.state.model_dir = Path(f"models/camembert_models")

@app.post("/")
async def predict(photo_type : str = Form(...),curiosity_level :int = Form(...), image_array : UploadFile |None =None, isbn: str | None = Form(None)):
    image = None
    # file download
    if image_array :
        image_bytes = await image_array.read()
        image= np.array(Image.open(io.BytesIO(image_bytes)), dtype=np.uint8)
        print(image)
    print("curiosity_level : " + str(curiosity_level))
    if image is not None :
        return main_pipeline(image,
                             photo_type,
                             curiosity_level,
                             app.state.model_dir,
                             app.state.dataset_path,
                             n_neighbors=3,
                             embeddings_sources=app.state.embeddings_sources,
                             alpha=app.state.alpha)
    else :
        print(isbn)
        return main_pipeline(isbn,
                             photo_type,
                             curiosity_level,
                             app.state.model_dir,
                             app.state.dataset_path,
                             n_neighbors=3,
                             embeddings_sources=app.state.embeddings_sources,
                             alpha=app.state.alpha)
