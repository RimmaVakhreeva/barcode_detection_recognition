from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from detection.detect import detect_barcodes

app = FastAPI()


@app.post("/scan/")
async def scan_barcodes(file: UploadFile = File(...)):
    # Load the image from the file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Use pyzbar to decode the barcode
    barcodes = detect_barcodes(image)

    # Prepare the response data
    response_data = []
    for barcode in barcodes:
        response_data.append({
            "bbox": barcode.rect,
            "text": barcode.data.decode("utf-8")
        })

    return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)