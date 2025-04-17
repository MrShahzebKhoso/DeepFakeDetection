# Deepfake Detection API

This is a Flask-based API for detecting deepfake videos using a ViT-based model. The system allows users to upload videos, analyze them for manipulation, and generate reports.

## Features
- Upload video/image files for analysis
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
Go to Image Upload or click Check an Image on Home page. Upload the image and click Analyze Image. It will display the results there.

Go to Video Upload or click Check an Video on Home page. Upload the video and click Analyze Video. It will display the results there.


### 3. Download Report
Once the analysis is complete, you can download the report. After completion of analysis, click Download Report. It will download a detailed pdf to your local system.

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

