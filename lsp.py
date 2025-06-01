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

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/newindex.html")

MODEL = tf.keras.models.load_model(r"C:\Users\prati\Desktop\Mini_Project_2\plant_disease12.h5")
# print("Model input shape:", MODEL.input_shape)


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

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))  # Change from 255 to 224
    image = np.array(image) / 255.0
    return image.astype(np.float32)

DISEASE_INFO = {
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "A fungal disease causing white powdery spots on cherry leaves and fruit, reducing photosynthesis and fruit quality.",
        "prevention": [
            "Plant resistant varieties.",
            "Prune trees to improve air circulation.",
            "Avoid overhead irrigation to keep foliage dry."
        ],
        "treatment": [
            "Apply fungicides such as sulfur or neem oil.",
            "Remove and destroy infected plant parts."
        ],
        "solutions": [
            "Maintain proper orchard sanitation.",
            "Use biological controls like beneficial fungi."
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "description": "A fungal disease causing reddish-brown pustules on maize leaves, leading to reduced photosynthesis and yield loss.",
        "prevention": [
            "Plant resistant maize hybrids.",
            "Rotate crops to reduce pathogen build-up.",
            "Control weeds which can harbor spores."
        ],
        "treatment": [
            "Apply fungicides like azoxystrobin at early disease stages.",
            "Use foliar fungicide sprays as needed."
        ],
        "solutions": [
            "Monitor weather conditions favorable for rust.",
            "Improve field drainage to reduce humidity."
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "A fungal disease causing large tan lesions with dark borders on maize leaves, affecting grain filling.",
        "prevention": [
            "Plant resistant varieties.",
            "Practice crop rotation.",
            "Remove crop residues after harvest."
        ],
        "treatment": [
            "Apply fungicides such as chlorothalonil or propiconazole.",
            "Timely spraying at the onset of symptoms."
        ],
        "solutions": [
            "Use disease forecasting models to time treatments.",
            "Ensure good nitrogen management to reduce stress."
        ]
    },
    "Corn_(maize)___healthy": {
        "description": "The maize plant shows no signs of disease and appears healthy.",
        "prevention": [
            "Maintain good agronomic practices.",
            "Monitor fields regularly for early disease detection."
        ],
        "treatment": [],
        "solutions": [
            "Use balanced fertilization.",
            "Ensure proper irrigation and weed control."
        ]
    },
    "Grape___Black_rot": {
        "description": "A fungal disease that causes black spots on leaves, stems, and fruit, leading to fruit rot and crop loss.",
        "prevention": [
            "Plant resistant grape varieties.",
            "Prune vines to increase air circulation.",
            "Remove mummified berries and infected plant debris."
        ],
        "treatment": [
            "Apply fungicides such as myclobutanil or sulfur.",
            "Regular spraying during the growing season."
        ],
        "solutions": [
            "Practice good vineyard sanitation.",
            "Use disease forecasting and monitoring tools."
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Causes irregular brown spots on grape leaves, leading to premature leaf drop and reduced photosynthesis.",
        "prevention": [
            "Remove infected leaves and debris.",
            "Ensure proper vine spacing for airflow.",
            "Avoid overhead irrigation."
        ],
        "treatment": [
            "Apply fungicides such as copper-based sprays.",
            "Use protective sprays during wet conditions."
        ],
        "solutions": [
            "Improve canopy management.",
            "Maintain balanced fertilization."
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "A bacterial disease causing water-soaked spots on pepper leaves and fruit, which may turn necrotic and reduce yield.",
        "prevention": [
            "Use certified disease-free seeds.",
            "Avoid overhead irrigation.",
            "Practice crop rotation."
        ],
        "treatment": [
            "Apply copper-based bactericides.",
            "Remove and destroy infected plant parts."
        ],
        "solutions": [
            "Maintain field hygiene.",
            "Use drip irrigation to reduce leaf wetness."
        ]
    },
    "Potato___Early_blight": {
        "description": "Fungal disease causing dark spots with concentric rings on potato leaves, reducing photosynthesis and yield.",
        "prevention": [
            "Plant resistant varieties.",
            "Rotate crops to prevent soilborne inoculum build-up.",
            "Use certified seed potatoes."
        ],
        "treatment": [
            "Apply fungicides such as chlorothalonil or mancozeb.",
            "Remove infected debris."
        ],
        "solutions": [
            "Improve drainage and reduce humidity in fields.",
            "Practice good weed control."
        ]
    },
    "Potato___Late_blight": {
        "description": "Serious fungal disease causing dark lesions on leaves and tubers, leading to crop failure if uncontrolled.",
        "prevention": [
            "Plant certified disease-free seed tubers.",
            "Avoid overhead irrigation.",
            "Remove infected plants promptly."
        ],
        "treatment": [
            "Use systemic fungicides like metalaxyl.",
            "Apply fungicides preventatively under favorable conditions."
        ],
        "solutions": [
            "Monitor weather and use forecasting models.",
            "Implement drip irrigation."
        ]
    },
    "Strawberry___Leaf_scorch": {
        "description": "A fungal disease causing leaf edges to dry and brown, reducing photosynthetic area.",
        "prevention": [
            "Use disease-free planting material.",
            "Avoid overhead irrigation.",
            "Space plants to improve air flow."
        ],
        "treatment": [
            "Apply fungicides such as captan or thiophanate-methyl.",
            "Remove and destroy infected leaves."
        ],
        "solutions": [
            "Maintain good soil fertility.",
            "Mulch plants to reduce soil splash."
        ]
    },
    "Tomato___Early_blight": {
        "description": "Fungal disease causing concentric brown spots on tomato leaves and fruit, leading to defoliation and yield loss.",
        "prevention": [
            "Use resistant varieties.",
            "Practice crop rotation.",
            "Use disease-free seed."
        ],
        "treatment": [
            "Apply fungicides like chlorothalonil or mancozeb.",
            "Remove and destroy infected plant material."
        ],
        "solutions": [
            "Improve field sanitation.",
            "Ensure proper plant spacing."
        ]
    },
    "Tomato___Leaf_Mold": {
        "description": "Fungal disease causing yellow spots on upper leaf surfaces and fuzzy mold growth underneath.",
        "prevention": [
            "Use resistant cultivars.",
            "Avoid excessive humidity and poor air circulation.",
            "Water plants at the base."
        ],
        "treatment": [
            "Apply fungicides such as copper-based products or chlorothalonil.",
            "Remove infected leaves."
        ],
        "solutions": [
            "Improve ventilation in greenhouses or fields.",
            "Practice good watering practices."
        ]
    }
}

@app.post("/product")
async def product(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Get additional information
        info = DISEASE_INFO.get(predicted_class, {})
        fone = 0
        
        if confidence < 0.6:
            return {
            'class': predicted_class,
            'confidence': 100 * float(confidence),
            'true class': "Unknown/Invalid",
            'F-1 Score' : fone,
            'description': info.get("description", "No description available."),
            'prevention': info.get("prevention", []),
            'treatment': info.get("treatment", []),
            "Solutions": info.get("solutions", [])
            }
        else:
            return {
            'class': predicted_class,
            'confidence': 100 * float(confidence),
            'true class': predicted_class,
            'F-1 Score' : 1,
            'description': info.get("description", "No description available."),
            'prevention': info.get("prevention", []),
            'treatment': info.get("treatment", []),
            "Solutions": info.get("solutions", [])
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)