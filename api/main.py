from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BATCH_SIZE, IMAGE_SIZE, CHANNELS = [32, 256, 3]
logging.basicConfig(level=logging.INFO)

try:
    MODEL = tf.keras.models.load_model("../Models/4")
except Exception as e:
    logging.error(f"Error loading the model: {str(e)}")
    raise HTTPException(status_code=500, detail="Internal Server Error")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
expected_input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
confidence_threshold = 0.5


@app.get("/ping")
async def ping():
    return "Hello, I am Alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        if image.shape != expected_input_shape[1:]:
            raise HTTPException(status_code=400, detail="Invalid image dimensions.")
    except HTTPException as e:
        # Handle HTTPExceptions explicitly and return detailed error responses
        return {"error": e.detail}
    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    image_batch = np.expand_dims(image, 0)

    try:
        prediction = MODEL.predict(image_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])

        if confidence < confidence_threshold:
            predicted_class = "Uncertain"

        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
