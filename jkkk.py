from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/")
# def read_root():
#     return FileResponse("static/index.html")

MODEL = tf.keras.models.load_model(r"C:\Users\prati\Desktop\Mini_Project_2\plant_disease12.h5")

CLASS_NAMES = [
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Pepper,_bell___Bacterial_spot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Strawberry___Leaf_scorch",
    "Tomato___Early_blight",
    "Tomato___Leaf_Mold"
]

true_class = [
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Pepper,_bell___Bacterial_spot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Strawberry___Leaf_scorch",
    "Tomato___Early_blight",
    "Tomato___Leaf_Mold"
]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    """ Convert uploaded file to a NumPy array (preprocessed image) """
    try:
        image = Image.open(BytesIO(data)).convert("RGB")  # Convert to RGB
        image = image.resize((224, 224))  # Resize to match model input size
        return np.array(image) / 255.0  # Normalize pixel values (0-1)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

from sklearn.metrics import f1_score

@app.route("/product", methods=["POST"])
def product():
    try:
        file = request.files['file']
        image = read_file_as_image(file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Optional: true label (for F1 score calculation)
        true_class = request.form.get("true_class")

        # Compute F1 score if true_class is provided and valid
        if true_class and true_class in CLASS_NAMES:
            y_true = [CLASS_NAMES.index(true_class)]
            y_pred = [np.argmax(predictions[0])]
            fone = f1_score(y_true, y_pred, average='macro')
        else:
            fone = 0.0

        info = DISEASE_INFO.get(predicted_class, {})

        result = {
            'class': predicted_class,
            'confidence': 100 * float(confidence),
            'true class': true_class if true_class else "Unknown/Invalid",
            'F-1 Score': float(fone),
            'description': info.get("description", "No description available."),
            'prevention': info.get("prevention", []),
            'treatment': info.get("treatment", []),
            'Solutions': info.get("solutions", [])
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)