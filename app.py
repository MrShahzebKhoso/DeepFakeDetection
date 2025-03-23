from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
from werkzeug.utils import secure_filename
import datetime
import json
from fpdf import FPDF
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['REPORT_FOLDER'] = 'reports/'
app.config['FRAMES_FOLDER'] = 'frames/'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAMES_FOLDER'], exist_ok=True)

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

def generate_report(filename, duration, fps, frames, processing_time):
    report_data = {
        "filename": filename,
        "timestamp": str(datetime.datetime.now()),
        "status": "Analysis Complete",
        "deepfake_detected": "Yes",  # Dummy output
        "video_duration": f"{duration} seconds",
        "fps": fps,
        "total_frames": frames,
        "processing_time": f"{processing_time} seconds"
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
    duration = random.randint(5, 60)
    fps = random.choice([24, 30, 60])
    frames = duration * fps
    processing_time = 2
    report_path = generate_report(filename, duration, fps, frames, processing_time)
    return jsonify({'message': 'Analysis complete', 'deepfake_detected': 'Yes', 'report': report_path})

@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename + ".pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
