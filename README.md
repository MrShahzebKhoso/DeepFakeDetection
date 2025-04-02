# Deepfake Detection API

This is a Flask-based API for detecting deepfake videos using a ViT-based model. The system allows users to upload videos, analyze them for manipulation, and generate reports.

## Features
- Upload video files for analysis
- Extract frames from videos
- Detect deepfake content using a Vision Transformer (ViT) model
- Generate reports in PDF format
- Download analysis reports
- RESTful API for easy integration

## Demo
A demonstration of the system is available in `assets/demo.mp4`.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install flask torch torchvision timm opencv-python pillow fpdf werkzeug
```

## Folder Structure
```
project_root/
│── app.py
│── uploads/          # Uploaded videos
│── reports/          # Generated reports
│── frames/           # Extracted frames from videos
│── model/
│   └── best_vit_model.pth  # Pretrained ViT model
│── templates/
│   └── index.html    # Frontend template
│── static/
│── demo.mp4          # Demo video
│── README.md
```

## Usage

### 1. Start the Flask Server
Run the application with:
```bash
python app.py
```
The server will start at `http://127.0.0.1:5000/`.



### Web UI
You can also use the Flask-based web UI for uploading and analyzing videos. The UI is designed with HTML, CSS, and JavaScript for a smooth user experience.


### 2. Upload a Video
Use the `/upload` endpoint to upload a video.
```bash
curl -X POST -F "file=@path/to/video.mp4" http://127.0.0.1:5000/upload
```

### 3. Analyze a Video
Trigger analysis on an uploaded video:
```bash
curl -X GET http://127.0.0.1:5000/analyze/video.mp4
```

### 4. Download Report
Download the generated PDF report:
```bash
curl -X GET http://127.0.0.1:5000/download_report/video.mp4 -o report.pdf
```

## API Endpoints
| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET`  | `/` | Home page |
| `POST` | `/upload` | Upload a video file |
| `GET`  | `/frames/<filename>/<frame>` | Retrieve an extracted frame |
| `GET`  | `/analyze/<filename>` | Analyze a video for deepfake detection |
| `GET`  | `/download_report/<filename>` | Download the analysis report |

## Model
The system uses a pre-trained Vision Transformer (ViT) model (`vit_large_patch16_224`) fine-tuned for deepfake detection. The model expects input images of size 224x224 with normalization.

## Acknowledgments
- [Timm Library](https://github.com/rwightman/pytorch-image-models) for the ViT model.
- [OpenCV](https://opencv.org/) for video processing.

## License
This project is licensed under the MIT License.

