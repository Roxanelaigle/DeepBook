import numpy as np
import easyocr
import matplotlib.pyplot as plt
import cv2

reader = easyocr.Reader(['fr','en']) # this needs to run once to load the model into memory

def picture_reader(img: np.ndarray) -> list[str]:
    '''
    Text detection & extraction: returns the text of a book cover, under the form of a list of string.
    '''
    text = reader.readtext(img, detail = 0)
    return text

def bboxes_drawer(img: np.ndarray) -> None:
    '''
    Displays the image with the bounding boxes that were detected during the text detection phase.
    '''
    result = reader.readtext(img, detail=1)  # Keep detail=1 to get bbox but ignore text

    # Loop through results and draw bounding boxes
    for bbox, _, _ in result:  # Ignore text and probability
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    # Convert to RGB and display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show();

if __name__ == "__main__":
    pass
