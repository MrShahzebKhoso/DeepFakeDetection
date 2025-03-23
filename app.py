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
import random
import time

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

# Define image transformations (same as used during training)
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    # total_frames might be useful if needed, but here we count frames processed
    processed_frames = 0
    real_count = 0
    manipulated_count = 0

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frames += 1
        # Convert frame to PIL Image and apply transformations
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        
        if predicted.item() == 0:
            real_count += 1
        else:
            manipulated_count += 1

    processing_time = time.time() - start_time
    cap.release()
    
    result = "Real" if real_count > manipulated_count else "Manipulated"
    return result, fps, processed_frames, processing_time

def extract_frames(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    frame_dir = os.path.join(app.config['FRAMES_FOLDER'], filename)
    os.makedirs(frame_dir, exist_ok=True)
    frame_count = 0
    frame_list = []
    while frame_count < 10:  # Extract first 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(frame_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_list.append(frame_filename)
        frame_count += 1
    cap.release()
    return frame_count, filename, frame_list

def generate_report(filename, duration, fps, frames, processing_time, detection_result):
    report_data = {
        "filename": filename,
        "timestamp": str(datetime.datetime.now()),
        "status": "Analysis Complete",
        "deepfake_detected": detection_result,
        "video_duration": f"{duration:.2f} seconds",
        "fps": fps,
        "total_frames": frames,
        "processing_time": f"{processing_time:.2f} seconds"
    }
    pdf_report_path = os.path.join(app.config['REPORT_FOLDER'], filename + ".pdf")
    
    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Deepfake Detection Report", ln=True, align='C')
    pdf.ln(10)
    for key, value in report_data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.output(pdf_report_path)
    
    return pdf_report_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        frame_count, frame_folder, frame_list = extract_frames(filepath, filename)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename, 'frames': frame_list})

@app.route('/frames/<filename>/<frame>', methods=['GET'])
def get_frame(filename, frame):
    frame_dir = os.path.join(app.config['FRAMES_FOLDER'], filename)
    return send_from_directory(frame_dir, frame)

@app.route('/analyze/<filename>', methods=['GET'])
def analyze_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Use the prediction function to determine if video is real or manipulated,
    # and get the actual metrics from processing the video.
    detection_result, fps, processed_frames, processing_time = predict_video(filepath, model, transform, device)
    
    # Calculate video duration from processed frames and fps
    duration = processed_frames / fps if fps > 0 else 0
    
    report_path = generate_report(filename, duration, fps, processed_frames, processing_time, detection_result)
    
    return jsonify({'message': 'Analysis complete', 'deepfake_detected': detection_result, 'report': report_path})

@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename + ".pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
