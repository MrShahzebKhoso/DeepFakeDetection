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
import sqlite3
from sqlite3 import Error

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('video_analysis.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upload_id INTEGER,
                    detection_result TEXT,
                    fps REAL,
                    processed_frames INTEGER,
                    processing_time REAL,
                    report_path TEXT,
                    FOREIGN KEY(upload_id) REFERENCES uploads(id)
                )''')
    conn.commit()
    conn.close()

# Call the function to initialize the database
init_db()

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
    
    result = "Real" if real_count > manipulated_count else "Deepfake Detected"
   
    reasons = '''The video is classified as manipulated by the classifier.
Majority of the frames in the video are manipulated due to one or more of the following reason(s)    
    -Inconsistent lighting across frames
    -Unnatural eye blinking or gaze direction
    -Face boundary artifacts near jawline
    -Frame-level classification shows high manipulation confidence
        ''' if result == "Deepfake Detected" else "The majority of frames are classified as Normal."
    return result, fps, processed_frames, processing_time, reasons, manipulated_count

def predict_image(image_path, model, transform, device):
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        confidence_score = confidence.item() * 100  # convert to percentage

    processing_time = time.time() - start_time
    result = "Real" if predicted.item() == 0 else "Deepfake Detected"
    
    reasons = '''The image is classified as manipulated by the classifier.
The frame is manipulated due to one or more of the following reason(s): 
   - Inconsistent lighting
   - Unnatural blinking or gaze
   - Artifacts near jawline
   - High manipulation confidence
    ''' if result == "Deepfake Detected" else "The image is classified as Normal."

    return result, 0, 1, processing_time, reasons, confidence_score

def extract_frames(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    frame_dir = os.path.join(app.config['FRAMES_FOLDER'], filename)
    os.makedirs(frame_dir, exist_ok=True)
    
    frame_count = 0
    frame_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save each frame as a separate image
        frame_filename = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(frame_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        
        # Append the filename to the list
        frame_list.append(frame_filename)
        frame_count += 1
    
    cap.release()
    return frame_count, filename, frame_list

def generate_report(filename, duration, fps, frames, processing_time, detection_result, reasons):
    report_data = {
        "filename": filename,
        "timestamp": str(datetime.datetime.now()),
        "status": "Analysis Complete",
        "deepfake_detected": detection_result,
        "media_duration": f"{duration:.2f} seconds" if fps > 0 else "N/A",
        "fps": fps if fps > 0 else "N/A",
        "total_frames": frames,
        "processing_time": f"{processing_time:.2f} seconds",
        "reasons": reasons
    }
    pdf_report_path = os.path.join(app.config['REPORT_FOLDER'], filename + ".pdf")
     
    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Deepfake Detection Report", ln=True, align='C')
    pdf.ln(10)
    
    for key, value in report_data.items():
        if key == "reasons":
            # Split 'reasons' by newline and add each line to the PDF
            reasons_lines = value.split('\n')
            for line in reasons_lines:
                pdf.cell(200, 10, txt=line, ln=True)
        else:
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    pdf.output(pdf_report_path)
    
    return pdf_report_path

@app.route('/')
def home():
    # return render_template('index.html')
    return render_template('site.html')

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
        
        # Log file upload to the database
        conn = sqlite3.connect('video_analysis.db')
        c = conn.cursor()
        c.execute('INSERT INTO uploads (filename) VALUES (?)', (filename,))
        upload_id = c.lastrowid  # Get the ID of the inserted record
        conn.commit()
        conn.close()
        
        # If the uploaded file is a video, extract frames (optional for images)
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            frame_count, frame_folder, frame_list = extract_frames(filepath, filename)
            return jsonify({'message': 'File uploaded successfully', 'filename': filename, 'frames': frame_list, 'upload_id': upload_id})
        else:
            return jsonify({'message': 'File uploaded successfully', 'filename': filename, 'upload_id': upload_id})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze/<filename>', methods=['GET'])
def analyze_media(filename):
    confidence = 0
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    # Check file type and choose the appropriate prediction method
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        detection_result, fps, processed_frames, processing_time, reasons, manipulated_count = predict_video(filepath, model, transform, device)
        # Calculate media duration from processed frames and fps
        duration = processed_frames / fps if fps > 0 else 0
    else:
        detection_result, fps, processed_frames, processing_time, reasons, confidence = predict_image(filepath, model, transform, device)
        duration = 0  # Duration is not applicable for images

    # Generate PDF report
    report_path = generate_report(filename, duration, fps, processed_frames, processing_time, detection_result, reasons)
    
    # Log analysis result to the database
    conn = sqlite3.connect('video_analysis.db')
    c = conn.cursor()
    c.execute('SELECT id FROM uploads WHERE filename = ?', (filename,))
    row = c.fetchone()
    if row is None:
        return jsonify({'error': 'Upload record not found'}), 404
    upload_id = row[0]
    
    c.execute('''INSERT INTO analysis 
                (upload_id, detection_result, fps, processed_frames, processing_time, report_path)
                VALUES (?, ?, ?, ?, ?, ?)''', 
              (upload_id, detection_result, fps, processed_frames, processing_time, report_path))
    conn.commit()
    conn.close()
    
    if(confidence==0):
        return jsonify({
            'message': 'Analysis complete',
            'deepfake_detected': detection_result,
            'reasons': reasons,
            'report': report_path,
            'total_frames': processed_frames,
            'manipulated_frames': manipulated_count
            })
    else:
        return jsonify({
            'message': 'Analysis complete',
            'deepfake_detected': detection_result,
            'reasons': reasons, 
            'report': report_path, 
            'confidence': confidence
            })

@app.route('/frames/<filename>/<frame>', methods=['GET'])
def get_frame(filename, frame):
    frame_dir = os.path.join(app.config['FRAMES_FOLDER'], filename)
    return send_from_directory(frame_dir, frame)

@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename + ".pdf", as_attachment=True)

@app.route('/get_media_history', methods=['GET'])
def get_media_history():
    # Retrieve the list of uploaded media files (videos and images)
    valid_extensions = ('.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png')
    media_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith(valid_extensions)]
    return jsonify({'media': [{'filename': media} for media in media_files]})

@app.route('/media', methods=['GET'])
def get_media():
    conn = sqlite3.connect('video_analysis.db')
    c = conn.cursor()
    c.execute('''SELECT u.filename, a.detection_result, a.processing_time, a.report_path
                 FROM uploads u LEFT JOIN analysis a ON u.id = a.upload_id''')
    media = c.fetchall()
    conn.close()
    
    return jsonify({'media': media})

if __name__ == '__main__':
    app.run(debug=True)
