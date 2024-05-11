from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from detection.detect import Yolov9
from ocr.model import OcrModel

app = FastAPI()
detection_model = Yolov9(weights=Path("./best.pt"))
ocr_model = OcrModel(weights=Path("./crnn_last.pt"))


@app.post("/scan/")
async def scan_barcodes(file: UploadFile = File(...)):
    image_data = await file.read()
    image = np.array(Image.open(io.BytesIO(image_data)))

    bboxes = detection_model.detect_barcodes(image)
    texts = ocr_model.recognize(image, bboxes)

    response_data = []
    for bbox, text in zip(bboxes, texts):
        response_data.append({
            "bbox": bbox[:4].tolist(),
            "bbox_confidence": bbox[4].tolist(),
            "text": text
        })

    return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
