import numpy as np
import easyocr

reader = easyocr.Reader(['fr','en']) # this needs to run once to load the model into memory

def converter(img: np.ndarray) -> list[str]:
    text = reader.readtext(img, detail = 0)
    return text

if __name__ == "__main__":
    pass
