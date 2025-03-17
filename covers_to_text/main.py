import easyocr
import numpy as np
import os
import unicodedata
import supervision as sv
from ultralytics import YOLO

# Constants

YOLO_MODEL = "yolo12l"
DEFAULT_PATH_YOLO_MODEL = os.path.join("models","yolo_models",YOLO_MODEL+".pt")

# Pre-loading -- running once to load into memory
# Note: YOLO then reloads each time thereafter

reader = easyocr.Reader(['fr','en']) # this needs to run once to load the model into memory
model = YOLO(DEFAULT_PATH_YOLO_MODEL)

def yolo_book_scanner(img: np.ndarray, show_output: bool = False) -> list[np.ndarray]:
    '''
    Tool for identifying & separating multiple books from a single picture, to be processed separately afterwards.
    The function looks for all the books present in a picture, and returns them as a list of cropped images (as numpy ndarrays).
    Books should either show their covers (e.g., multiple books laid out on a table) or show their sides (e.g., row of books on a bookshelf).
    '''

    # init; default model: yolo 12l -> best balance found between speed & accuracy
    model = YOLO(DEFAULT_PATH_YOLO_MODEL)

    # running model on the image
    # only looking for class #73 (books)
    # iou decreased to 0.1, i.e., low tolerance to duplicates/overlaps
    results = model.predict(img, classes=[73], iou= 0.1)

    # getting coordinates of each book
    cropped_books = [img[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in results[0].boxes.xyxy.numpy()]

    # error management: in case of failure of the YOLO-based book detection (i.e., no book identified at all)
    if len(cropped_books) == 0:
        if show_output == True:
            print(f"⚠️ Picture has been scanned, but no book was detected. Returning original image as output.")
            sv.plot_image(img, size=(7,7))
        return [img]

    # optional: display the output with the bounding boxes
    if show_output == True:

        # init
        annotated_img = img.copy()

        # getting bounding boxes
        detections = sv.Detections(
            xyxy = results[0].boxes.xyxy.numpy(),
            confidence=results[0].boxes.conf.numpy(),
            class_id=results[0].boxes.cls.numpy().astype(int)
        )

        # bboxes drawing
        annotator = sv.BoxAnnotator(thickness=2)
        annotated_img = annotator.annotate(annotated_img, detections)

        # bboxes labels writing
        labels=[results[0].names.get(cls, str(cls)) for cls in results[0].boxes.cls.numpy().astype(int)]
        annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
        annotated_img = annotator.annotate(annotated_img, detections, labels=labels)

        # ploting image
        print(f"✅ Picture has been scanned. {len(cropped_books)} book(s) have been identified in the picture.")
        sv.plot_image(annotated_img, size=(7,7))

    # making the scans horizontal in the case the books were scanned vertically (useful for pictures of spines)
    # rule used: vertical length should be > 3 * horizontal length
    v_dim_avg = np.average([book.shape[0] for book in cropped_books])
    h_dim_avg = np.average([book.shape[1] for book in cropped_books])
    if v_dim_avg > 3*h_dim_avg:
        cropped_books = [np.rot90(book, k=-1) for book in cropped_books]

    return cropped_books

def picture_reader(img: np.ndarray, show_output: bool = False) -> list[str]:
    '''
    Text detection & extraction: returns the text of a book cover, under the form of a list of string.
    Uses easy0CR without fine-tuning.
    '''
    results = reader.readtext(img, detail = 1)

    # filtering out which text blocks to keep or not, based on the OCR's confidence level
    # + extracting the text as a list of strings
    keep_or_not = []
    text_chunks = []
    for result in results:
        if result[2] >= 0.25:
            keep_or_not.append(True)
            text_chunks.append(result[1])
        else:
            keep_or_not.append(False)

    # cleaning up our list of strings
    punctuation_list = """$€£"+°!#%()*+,./:;<=>?@[\]^_`{|}~"""
    for i in range(len(text_chunks)):
        chunk = text_chunks[i]
        chunk = unicodedata.normalize('NFKD', chunk).encode('ASCII', 'ignore').decode('utf-8') # accents removal
        for p in punctuation_list:
            chunk = chunk.replace(p,'') # punctuation & symbols removal
        chunk = chunk.lower() # lowercase
        chunk = " ".join(chunk.split()) # strip
        text_chunks[i] = chunk

    # error management: in case of failure of the OCR (i.e., no text at all)
    if len(text_chunks) == 0:
        if show_output == True:
            print(f"⚠️ Picture has been scanned, but no text was detected. Showing original image instead.")
            sv.plot_image(img, size=(7,7))
        return text_chunks

    # optional: display the output with the bounding boxes
    if show_output == True:
        # init
        annotated_img = img.copy()

        # defining colors & labels
        colors = np.array([9 if keep else 1 for keep in keep_or_not]) # trick: using supervision's class_id default colors
        labels = [str(int(proba*100))+"%" for _,_,proba in results]

        # getting bounding boxes coordinates in the right format for supervision
        xyxy = []
        for bbox,_,_ in results:
            xyxy.append([bbox[0][0],bbox[0][1],bbox[2][0],bbox[2][1]])
        xyxy = np.array(xyxy)
        detections = sv.Detections(xyxy = xyxy,class_id=colors)

        # bboxes drawing
        annotator = sv.BoxAnnotator(thickness=2)
        annotated_img = annotator.annotate(annotated_img, detections)

        # bboxes labels writing
        annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
        annotated_img = annotator.annotate(annotated_img, detections, labels=labels)

        # ploting image
        print(f"✅ Picture has been OCRised. {len(text_chunks)} valid block(s) of texts have been successfully identified.")
        print(text_chunks)
        sv.plot_image(annotated_img, size=(7,7))

    return text_chunks

def picture_reader_multibooks(img: np.ndarray, show_output: bool = False) -> list[list[str]]:
    '''
    1. Applies YOLO to cut the initial images into multiple sub-images each containing 1 book.
    2. Applies picture_reader on each sub-image (1 book = 1 list of strings).
    3. Returns a list of list of strings.
    '''
    if show_output == True:
        print(f"➡️ Applying YOLO to identify all the books of the picture input.")
    imgs = yolo_book_scanner(img,show_output)
    result = []
    if show_output == True:
        print(f"➡️ Reading the text of {len(imgs)} books identified thanks to the YOLO model.")
    for i in imgs:
        result.append(picture_reader(i,show_output))
    return result

if __name__ == "__main__":
    pass
