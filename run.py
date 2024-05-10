from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from detection.detect import Yolov9

app = FastAPI()
detection_model = Yolov9(weights=Path("./best.pt"))


@app.post("/scan/")
async def scan_barcodes(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    bboxes = detection_model.detect_barcodes(np.array(image))

    response_data = []
    for bbox in bboxes:
        response_data.append({
            "bbox": bbox[:4].tolist(),
            "bbox_confidence": bbox[4].tolist(),
            # "text": barcode.data.decode("utf-8")
        })

    return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
