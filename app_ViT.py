from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from werkzeug.utils import secure_filename
import datetime
import json
from fpdf import FPDF

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['REPORT_FOLDER'] = 'reports/'
app.config['FRAMES_FOLDER'] = 'frames/'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAMES_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=2)
model.load_state_dict(torch.load('model/best_vit_model.pth', map_location=device))
model.to(device)
model.eval()

def predict_video(video_path, model, transform, device):
    cap = cv2.VideoCapture(video_path)
    real_count = 0
    manipulated_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        
        if predicted.item() == 0:
            real_count += 1
        else:
            manipulated_count += 1
    
    cap.release()
    return "Real" if real_count > manipulated_count else "Manipulated"

@app.route('/analyze/<filename>', methods=['GET'])
def analyze_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    result = predict_video(filepath, model, transform, device)
    report_path = os.path.join(app.config['REPORT_FOLDER'], filename + ".pdf")
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Deepfake Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Filename: {filename}", ln=True)
    pdf.cell(200, 10, txt=f"Detection Result: {result}", ln=True)
    pdf.output(report_path)
    
    return jsonify({'message': 'Analysis complete', 'deepfake_detected': result, 'report': report_path})

if __name__ == '__main__':
    app.run(debug=True)
