
from fastapi import FastAPI, Request
import json
import numpy as np

app = FastAPI()

@app.post("/")
async def predict(request: Request):

    raw_data = await request.body()
    data=json.loads(raw_data)
    curiosity_level = data['curiosity_level']
    if 'image_array' in data.keys() :
        image_array = np.array(data['image_array'])
        return full_model(image_array, curiosity_level)
    else :
        isbn = data['isbn']
        return full_model(isbn, curiosity_level)
