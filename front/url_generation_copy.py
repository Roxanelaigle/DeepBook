from fastapi import FastAPI, Request
from typing import List, Any
import json
import numpy as np

app = FastAPI()

# Endpoint POST avec le modèle
@app.post("/")
async def predict(request: Request):

    raw_data = await request.body()
    data=json.loads(raw_data)
    # payload = {"image_array": img_array_rgb.tolist(), "curiosity_level": curiosity_level}
    curiosity_level = data['curiosity_level']
    if 'image_array' in data.keys() :
        image_array = np.array(data['image_array'])
        return {'input_book' :
        {'title': 'xx' , 'authors': 'xx', 'image': 'xx', 'isbn': '9782401118522'},
        'output_books' :
          [{'title': 'xx' , 'authors': 'xx', 'image': 'xx', 'isbn': '9782401118522' }]}

    else :
        isbn = data['isbn']
        return {'input_book' :
        {'title': 'xx' , 'authors': 'xx', 'image': 'xx', 'isbn': isbn },
        'output_books' :
          [{'title': 'xx' , 'authors': 'xx', 'image': 'xx', 'isbn': isbn }]}
