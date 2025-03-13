
from fastapi import FastAPI, Request
import json
import numpy as np
from main_pipeline.main import main_pipeline

app = FastAPI()

@app.post("/")
async def predict(request: Request):

    raw_data = await request.body()
    data=json.loads(raw_data)
    curiosity_level = data['curiosity_level']
    if 'image_array' in data.keys() :
        image_array = np.array(data['image_array'])
        return main_pipeline(image_array, curiosity_level)
    else :
        isbn = data['isbn']
        return main_pipeline(isbn, curiosity_level)
