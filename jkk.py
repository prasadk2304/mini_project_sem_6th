from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Load model
MODEL = tf.keras.models.load_model(r"C:\Users\prati\Desktop\Mini_Project_2\plant_disease12.h5")

CLASS_NAMES = [
    "Cherry (including sour) Powdery mildew",
    "Corn (maize) Common rust",
    "Corn (maize) Northern Leaf Blight",
    "Corn (maize) healthy",
    "Grape Black rot",
    "Grape Leaf blight (Isariopsis Leaf Spot)",
    "Pepper, bell Bacterial spot",
    "Potato Early blight",
    "Potato Late blight",
    "Strawberry Leaf scorch",
    "Tomato Early blight",
    "Tomato Leaf Mold"
]

# Disease info dictionary
DISEASE_INFO = {
    "Cherry (including sour) Powdery mildew": {
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
    "Corn (maize) Common rust": {
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
    "Corn (maize) Northern Leaf Blight": {
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
    "Corn (maize) healthy": {
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
    "Grape Black rot": {
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
    "Grape Leaf blight (Isariopsis Leaf Spot)": {
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
    "Pepper, bell Bacterial spot": {
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
    "Potato Early blight": {
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
    "Potato Late blight": {
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
    "Strawberry Leaf scorch": {
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
    "Tomato Early blight": {
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
    "Tomato Leaf Mold": {
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

# Utility function
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image.astype(np.float32)

@app.route("/")
def root():
    return send_from_directory("static", "unknown.html")

@app.route("/ping", methods=["GET"])
def ping():
    return "Hello, I am alive"

@app.route("/product", methods=["POST"])
def product():
    try:
        file = request.files['file']
        image = read_file_as_image(file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        info = DISEASE_INFO.get(predicted_class, {})
        fone = 0

        if confidence < 0.6:
            return jsonify({
                'class': predicted_class,
                'model_accuracy': 77.78,
                'true class': "Unknown/Invalid",
                # 'F-1 Score': fone,
                'description': info.get("description", "No description available."),
                'prevention': info.get("prevention", []),
                'treatment': info.get("treatment", []),
                'Solutions': info.get("solutions", [])
            })
        else:
            return jsonify({
                'class': predicted_class,
                'model_accuracy': 77.78,
                'true class': predicted_class,
                # 'F-1 Score': 1,
                'description': info.get("description", "No description available."),
                'prevention': info.get("prevention", []),
                'treatment': info.get("treatment", []),
                'Solutions': info.get("solutions", [])
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)
