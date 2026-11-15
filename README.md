# 🛡️ Real-Life Violence Detection System

An AI-powered video analysis application that detects violent and non-violent activities in video footage using a **MoBiLSTM deep learning model**. The project combines a FastAPI backend for video processing and prediction with a Streamlit frontend for user interaction.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-7B2DFF?logo=railway)](https://railway.com/)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Model Information](#-model-information)
- [Installation and Setup](#-installation-and-setup)
- [Environment Variables](#-environment-variables)
- [API Documentation](#-api-documentation)
- [Deployment on Railway](#-deployment-on-railway)
- [Troubleshooting](#-troubleshooting)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🚀 Project Overview

Violence detection in video surveillance is an important computer vision application. Manually monitoring large amounts of video footage can be time-consuming and inefficient.

This project uses deep learning and computer vision to analyze video sequences and classify activities into two categories:

- **Violence:** The model identifies visual patterns associated with violent activity.
- **Non-Violence:** The model identifies activity that is not classified as violent.

The application uses a MobileNet-based CNN and LSTM-based temporal learning approach (MoBiLSTM) to extract spatial and temporal information from video frames.

The system provides a web interface where users can upload videos and receive prediction results through a FastAPI-powered backend.

## ✨ Key Features

- 🎥 Video Upload: Upload video files for violence detection.
- 🧠 Deep Learning Prediction: Uses a trained MoBiLSTM model.
- 👁️ Computer Vision: Extracts and processes video frames.
- ⚡ FastAPI Backend: Provides REST API endpoints for predictions.
- 🌐 Streamlit Frontend: Provides an interactive web interface.
- 📊 Prediction Results: Displays the model's classification results.
- 🔌 REST API Integration: Connects the frontend and backend.
- ☁️ Cloud Deployment: Supports independent frontend and backend deployment on Railway.
- 📖 API Documentation: Interactive API documentation using Swagger UI.

## ⚙️ How It Works

The application follows this processing pipeline:

1. The user uploads a video through the Streamlit interface.
2. The frontend sends the video to the FastAPI backend.
3. The backend extracts frames from the uploaded video.
4. The frames are resized and prepared for model input.
5. The MoBiLSTM model analyzes the video sequence.
6. The model generates a prediction.
7. The backend returns the prediction response.
8. The frontend displays the result to the user.

### System Workflow

```text
          User
            |
            v
     Streamlit Frontend
            |
            v
       Video Upload
            |
            v
       FastAPI Backend
            |
            v
      Video Processing
            |
            v
       Frame Extraction
            |
            v
       Frame Preprocessing
            |
            v
       MoBiLSTM Model
            |
            v
     Prediction Generation
            |
            v
       API Response
            |
            v
    Display Prediction
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning framework |
| MoBiLSTM | Video classification model |
| MobileNet | Spatial feature extraction |
| LSTM | Temporal sequence learning |
| OpenCV | Video processing and frame extraction |
| NumPy | Numerical operations |
| FastAPI | Backend API development |
| Pydantic | Data validation |
| Streamlit | Frontend development |
| Uvicorn | ASGI application server |
| Railway | Cloud deployment |
| Git & GitHub | Version control and source code hosting |

## 📁 Project Architecture

```text
violence-detection/
│
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── video_utils.py
│   ├── model_service.py
│   └── config.py
│
├── frontend/
│   ├── app.py
│   └── .streamlit/
│       └── config.toml
│
├── models/
│   └── Real_Life_Violence_Detation_MoBiLSTM_model.h5
│
├── requirements-backend.txt
├── requirements-frontend.txt
│
├── run_backend.sh
├── run_frontend.sh
│
└── README.md
```

### Folder Description

| File / Folder | Description |
|---|---|
| `backend/main.py` | FastAPI application and API endpoints |
| `backend/schemas.py` | Request and response validation schemas |
| `backend/video_utils.py` | Video frame extraction and preprocessing |
| `backend/model_service.py` | Loads and manages the trained model |
| `backend/config.py` | Application configuration and prediction settings |
| `frontend/app.py` | Streamlit user interface |
| `models/` | Stores the trained deep learning model |
| `requirements-backend.txt` | Backend dependencies |
| `requirements-frontend.txt` | Frontend dependencies |

## 🧠 Model Information

This project uses a trained **MoBiLSTM (MobileNet + Bidirectional LSTM) model** for video-based violence detection.

The model processes video frames to learn spatial and temporal patterns.

### Model Input

| Parameter | Value |
|---|---|
| Model Type | MoBiLSTM |
| Framework | TensorFlow / Keras |
| Input Sequence Length | 16 frames |
| Image Height | 64 pixels |
| Image Width | 64 pixels |
| Color Channels | 3 (RGB) |
| Input Shape | `(16, 64, 64, 3)` |
| Model Format | `.h5` |

### Model Processing

- MobileNet-based feature extraction captures spatial information from video frames.
- LSTM-based sequence learning captures temporal information across frames.
- The trained model generates a classification prediction.

**Note:** This repository contains the trained model and inference application. Training accuracy and evaluation metrics should be added only after verifying them against the original model evaluation results.

## 💻 Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Deep07155/violence-detection.git
```

Navigate into the project directory:

```bash
cd violence-detection
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements-backend.txt
```

### 4. Install Frontend Dependencies

```bash
pip install -r requirements-frontend.txt
```

### 5. Run the Backend

Open the first terminal from the project root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Backend URL:

```text
http://localhost:8000
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

### 6. Run the Frontend

Open a second terminal from the project root and activate the virtual environment.

Set the backend URL.

**Windows PowerShell:**

```powershell
$env:API_URL="http://localhost:8000"
```

**Linux / macOS:**

```bash
export API_URL="http://localhost:8000"
```

Start Streamlit:

```bash
streamlit run frontend/app.py
```

Frontend URL:

```text
http://localhost:8501
```

Your application should now be accessible locally.

## 🔐 Environment Variables

The application supports the following configuration variables:

| Variable | Default Value | Description |
|---|---|---|
| `MODEL_PATH` | `models/Real_Life_Violence_Detation_MoBiLSTM_model.h5` | Path to the trained model |
| `MAX_UPLOAD_MB` | `200` | Maximum video upload size in MB |
| `MAX_SEGMENTS` | `8` | Maximum timeline segments |
| `API_URL` | `http://localhost:8000` | FastAPI backend URL used by the frontend |

For production deployment, configure the backend and frontend URLs using Railway environment variables.

## 🔌 API Documentation

The backend is developed using FastAPI and exposes endpoints for health checking and video prediction.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks backend and model health |
| POST | `/predict` | Accepts a video and generates a prediction |
| GET | `/docs` | Interactive Swagger API documentation |

### Health Check

```http
GET /health
```

This endpoint checks the backend's health and model loading status.

### Video Prediction

```http
POST /predict?threshold=0.5
```

The prediction endpoint accepts a video file through multipart form data.

Example using cURL:

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.5" \
  -F "file=@my_video.mp4"
```

You can also test the API using the interactive Swagger interface:

```text
http://localhost:8000/docs
```

## ☁️ Deployment on Railway

This project uses two separate Railway services:

1. **Backend Service:** FastAPI + TensorFlow + trained model.
2. **Frontend Service:** Streamlit web application.

Both services use the same GitHub repository but have separate build and start commands.

### 1. Deploy the Backend

Create a Railway project and deploy the GitHub repository:

```text
https://github.com/Deep07155/violence-detection
```

Configure the backend service:

| Setting | Configuration |
|---|---|
| Root Directory | `/` |
| Region | Southeast Asia (Singapore) |
| Custom Build Command | `pip install -r requirements-backend.txt` |
| Custom Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Target Port | `8080` (based on the current deployment) |

Generate a public domain for the backend.

### 2. Configure Backend Variables

Add the following environment variables to the backend service:

```env
MODEL_PATH=models/Real_Life_Violence_Detation_MoBiLSTM_model.h5
MAX_UPLOAD_MB=200
MAX_SEGMENTS=8
```

### 3. Deploy the Frontend

Create a second service in the same Railway project using the same GitHub repository.

Configure the frontend:

| Setting | Configuration |
|---|---|
| Root Directory | `/` |
| Region | Southeast Asia (Singapore) |
| Custom Build Command | `pip install -r requirements-frontend.txt` |
| Custom Start Command | `streamlit run frontend/app.py --server.address 0.0.0.0 --server.port $PORT` |
| Target Port | `8080` (based on the current deployment) |

Railway may assign a different port in another deployment. Use the port shown in the deployment logs when generating the service domain.

### 4. Connect Frontend and Backend

Open the frontend service's Variables tab.

Add:

```env
API_URL=https://YOUR-BACKEND-DOMAIN
```

Replace the example with your actual backend public domain.

Make sure the frontend reads this environment variable when sending prediction requests.

### 5. Access the Deployed Application

After successful deployment, your services will have separate public URLs.

**Frontend:**

```text
https://YOUR-FRONTEND-DOMAIN
```

**Backend:**

```text
https://YOUR-BACKEND-DOMAIN
```

**Backend API Documentation:**

```text
https://YOUR-BACKEND-DOMAIN/docs
```

---

## 🐛 Troubleshooting

### 1. Model Loading Error

If the model fails to load:

- Verify the model file exists inside the `models/` directory.
- Verify the `MODEL_PATH` environment variable.
- Check TensorFlow compatibility with the version used during model training.
- Review backend deployment logs.

### 2. Video Upload Error

If the video cannot be processed:

- Check the supported video format.
- Check the maximum upload size.
- Try re-encoding the video using FFmpeg.

Example:

```bash
ffmpeg -i input.mov -c:v libx264 output.mp4
```

### 3. Frontend Cannot Connect to Backend

Verify that:

- The backend service is running.
- The backend public domain is correct.
- The frontend `API_URL` variable contains the backend URL.
- The frontend uses `API_URL` instead of a hardcoded localhost address.

### 4. TensorFlow GPU Warning

If deployment logs show that CUDA drivers are unavailable, TensorFlow may use CPU instead.

GPU access is not required for the application to start, but CPU inference may be slower.

### 5. Railway Out-of-Memory Error

Video processing and TensorFlow inference can require significant memory.

Possible solutions:

- Use an appropriate memory allocation.
- Reduce video processing workload.
- Limit video upload size and duration.
- Optimize frame extraction and model inference.

---

## ⚠️ Limitations

- Prediction quality depends on the trained model and the input video.
- Processing speed depends on available CPU and memory resources.
- The application is designed for video-based inference, not guaranteed real-time surveillance.
- Model predictions can be incorrect and should not be treated as definitive evidence of violence.
- Performance may vary with video quality, lighting, camera angle and activity.

## 🔮 Future Improvements

- [ ] Improve model accuracy and generalization.
- [ ] Optimize inference speed.
- [ ] Add support for more video formats.
- [ ] Improve video processing and resource management.
- [ ] Add prediction history and analytics.
- [ ] Improve frontend design and user experience.
- [ ] Add more comprehensive model evaluation metrics.
- [ ] Explore real-time video stream analysis.

## 👨‍💻 Author

**Deep Panchal**

AI/ML Developer | Python Developer

- GitHub: [@Deep07155](https://github.com/Deep07155)
- Project Repository: [Violence Detection](https://github.com/Deep07155/violence-detection)

---

## ⭐ Support

If you find this project interesting or useful, consider giving the repository a ⭐ on GitHub.

Contributions, suggestions and feedback are welcome!

---

**Disclaimer:** This project is developed for educational, research and demonstration purposes. Its predictions should not be used as the sole basis for security decisions or legal judgments.
