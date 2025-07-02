from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
MODEL = tf.keras.models.load_model("../models/potato_disease_model_v1.h5")
CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]
@app.get("/ping")
async def ping():
    return "hello world"

def read_files_as_images(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
@app.post("/predict")
async def predict(
    file:UploadFile = File(...)
    ):
    image = read_files_as_images(await file.read())
    image_batch=  np.expand_dims(image, axis=0 )
    predictions = MODEL.predict(image_batch)
    predicted_calss = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        "predicted_class": predicted_calss,
        "confidence": float(confidence)
    }       




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)