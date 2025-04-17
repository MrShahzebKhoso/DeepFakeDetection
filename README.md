# 🧠 Deepfake Detection API using Vision Transformer (ViT) | Flask-Based Video Analysis

A high-performance **Flask API for Deepfake Detection** using **Vision Transformer (ViT)** models. This AI-powered system analyzes videos and images to detect manipulated (deepfake) content and generates detailed PDF reports. Ideal for researchers, developers, and security analysts working in deepfake forensics, computer vision, or AI model deployment.

---

## 🔍 Features of the Deepfake Detection System

- 📤 Upload **video** or **image** files for deepfake detection
- 🖼️ Automatically extract video frames
- 🧠 Analyze content using a fine-tuned **Vision Transformer (ViT)** deep learning model
- 📄 Generate and download detailed **PDF reports**
- 🌐 RESTful API endpoints for integration into other tools
- 🖥️ Clean and user-friendly **web UI** built with Flask

---

## 🎥 Demo Video

> See how the deepfake detection API works in action:

[▶️ Click to watch the demo video](assets/demo.mp4)

<!--
<video width="600" controls>
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
-->

---

## 📦 Requirements & Installation

Make sure Python and pip are installed. Then, install the required packages:

```bash
pip install flask torch torchvision timm opencv-python pillow fpdf werkzeug
```

---

## 📁 Folder Structure

```
project_root/
│── app.py                        # Main Flask application
│── uploads/                     # Uploaded video/image files
│── reports/                     # Generated PDF reports
│── frames/                      # Extracted video frames
│── model/
│   └── best_vit_model.pth       # Pretrained Vision Transformer model
│── templates/
│   └── index.html               # Frontend HTML template
│── static/                      # Static files (CSS, JS)
│── assets/
│   └── demo.mp4                 # Demo video of the system
│── README.md
```

---

## 🚀 How to Use

### 1. Start the Flask Server

```bash
python app.py
```

Visit your browser at: `http://127.0.0.1:5000/`

---

### 2. Use the Web UI

- Navigate to **Image Upload** or **Video Upload** tabs.
- Select and upload a file.
- Click **Analyze** to run deepfake detection.
- Results will be displayed immediately.

---

### 3. Download the Analysis Report

After analysis is complete:

- Click **Download Report**.
- A comprehensive **PDF report** will be generated and saved locally.

---

## 🔗 REST API Endpoints

| Method | Endpoint | Function |
|--------|----------|----------|
| `GET`  | `/` | Home page |
| `POST` | `/upload` | Upload a video or image |
| `GET`  | `/frames/<filename>/<frame>` | Retrieve an extracted frame |
| `GET`  | `/analyze/<filename>` | Analyze for deepfake content |
| `GET`  | `/download_report/<filename>` | Download PDF report |

---

## 🤖 Model Details

The system uses a **Vision Transformer** model (`vit_large_patch16_224`) from the [Timm Library](https://github.com/rwightman/pytorch-image-models). The model is fine-tuned for deepfake detection and expects 224x224 normalized inputs.

---

## 🙏 Acknowledgments

- [Timm Library](https://github.com/rwightman/pytorch-image-models) – Pretrained ViT models
- [OpenCV](https://opencv.org/) – Video and image processing

---

## ⚖️ License

This project is released under the [MIT License](LICENSE).

---

## 📢 Keywords

`deepfake detection`, `deepfake analysis`, `ViT model`, `Vision Transformer`, `Flask API`, `video forensics`, `AI video detection`, `deepfake API`, `computer vision`, `deep learning`, `transformer model`, `fake video detector`, `AI-powered security`, `python project`