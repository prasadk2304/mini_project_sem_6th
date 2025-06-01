from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from werkzeug.utils import secure_filename
from fpdf import FPDF

app = Flask(__name__)

# Load the trained model
MODEL_PATH = r"C:\Users\prati\Desktop\Mini_Project_2\plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
CLASS_NAMES = [
    "Apple Apple scab", "Apple Black rot", "Apple Cedar apple rust", "Apple healthy",
    "Blueberry healthy", "Cherry Powdery mildew", "Cherry healthy",
    "Corn Cercospora leaf spot Gray leaf spot", "Corn Common rust", "Corn Northern Leaf Blight",
    "Corn healthy", "Grape Black rot", "Grape Esca (Black Measles)", "Grape Leaf blight",
    "Grape healthy", "Orange Haunglongbing (Citrus greening)", "Peach Bacterial spot",
    "Peach healthy", "Pepper bell Bacterial spot", "Pepper bell healthy",
    "Potato Early blight", "Potato Late blight", "Potato healthy", "Raspberry healthy",
    "Soybean healthy", "Squash Powdery mildew", "Strawberry Leaf scorch", "Strawberry healthy",
    "Tomato Bacterial spot", "Tomato Early blight", "Tomato Late blight", "Tomato Leaf Mold",
    "Tomato Septoria leaf spot", "Tomato Spider mites", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Tomato mosaic virus", "Tomato healthy"
]

# Load plant disease info
with open("plant_info.json", "r") as file:
    PLANT_INFO = json.load(file)

UPLOAD_FOLDER = "uploads"
HEATMAP_FOLDER = "heatmaps"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return np.expand_dims(image, axis=0)


def generate_heatmap(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(HEATMAP_FOLDER, os.path.basename(image_path))
    cv2.imwrite(heatmap_path, heatmap)
    return heatmap_path


def calculate_severity(heatmap_path):
    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
    severity = np.sum(heatmap > 100) / heatmap.size * 100
    return round(severity, 2)


def generate_pdf_report(original_img, heatmap_img, disease, severity, info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Disease Report: {disease}", ln=True, align='C')
    pdf.cell(200, 10, f"Severity: {severity}%", ln=True, align='C')
    pdf.image(original_img, x=10, y=30, w=90)
    pdf.image(heatmap_img, x=110, y=30, w=90)
    pdf.ln(100)
    pdf.multi_cell(0, 10, json.dumps(info, indent=4))
    pdf_path = os.path.join("reports", f"{os.path.basename(original_img).split('.')[0]}.pdf")
    os.makedirs("reports", exist_ok=True)
    pdf.output(pdf_path)
    return pdf_path


@app.route("/predict", methods=["POST"])
def predict():
    print("Received request:", request.files)  # Debugging line
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Predict disease
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    disease = CLASS_NAMES[class_index]
    confidence = round(prediction[class_index] * 100, 2)

    # Generate heatmap
    heatmap_path = generate_heatmap(file_path)

    # Calculate severity
    severity = calculate_severity(heatmap_path)

    # Fetch disease information
    disease_info = PLANT_INFO.get(disease, {})

    # Generate PDF report
    pdf_path = generate_pdf_report(file_path, heatmap_path, disease, severity, disease_info)

    return jsonify({
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "pdf_report": pdf_path
    })


@app.route("/download_report", methods=["GET"])
def download_report():
    report_path = request.args.get("path")
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return jsonify({"error": "Report not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)