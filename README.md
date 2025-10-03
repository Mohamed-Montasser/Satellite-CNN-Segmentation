# 🛰️ Satellite Water Segmentation API

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0.0-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/PyTorch-2.2.0-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A deep learning-powered Flask API for water body segmentation in satellite imagery using U-Net architecture with MobileNetV2 encoder. This application processes 12-channel TIFF satellite images and generates binary water masks with high accuracy.

## 🌟 Features

- **Advanced Preprocessing**: Automatic generation of NDWI and NDVI indices from satellite bands
- **Deep Learning Model**: U-Net architecture with MobileNetV2 encoder for efficient segmentation
- **REST API**: Simple POST endpoints for programmatic access
- **Web Interface**: User-friendly upload form for testing and visualization
- **Real-time Processing**: Fast inference with PyTorch
- **Multi-format Output**: Base64 JSON response or downloadable PNG masks
- **High Performance**: Trained on satellite imagery with >90% accuracy

## 🎬 Demo

### Web Interface
Access the interactive web interface at `http://localhost:5000/upload_form`

![Demo Screenshot](screenshots/demo.png)

### API Usage
```
import requests

url = 'http://localhost:5000/predict'
files = {'image': open('satellite_image.tif', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## 🏗️ Model Architecture

**Architecture**: U-Net with MobileNetV2 Encoder
- **Input Channels**: 11 (after feature engineering)
- **Output**: Binary segmentation mask
- **Loss Function**: Combined Dice Loss + BCE Loss (0.5:0.5)
- **Optimizer**: Adam (lr=1e-3)
- **Metrics**: IoU (Intersection over Union), Binary Accuracy

### Feature Engineering
The model performs automatic feature extraction:
1. Normalizes all 12 input channels
2. Generates **NDWI** (Normalized Difference Water Index): `(Green - NIR) / (Green + NIR)`
3. Generates **NDVI** (Normalized Difference Vegetation Index): `(NIR - Red) / (NIR + Red)`
4. Removes less informative channels (Blue, Green, Red)
5. Final input: 11 channels

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- CUDA-compatible GPU (optional, for faster inference)

### Step 1: Clone the Repository
```
git clone https://github.com/yourusername/satellite-segmentation-api.git
cd satellite-segmentation-api
```

### Step 2: Create Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```
pip install -r requirements.txt
```

**Important**: If you encounter NumPy compatibility issues, run:
```
pip uninstall numpy
pip install "numpy<2"
```

### Step 4: Download Model Weights
Place the trained model file `best_model.pth` in the project root directory.

### Step 5: Run the Application
```
python app.py
```

The API will be available at `http://localhost:5000`

## 📖 Usage

### Web Interface
1. Navigate to `http://localhost:5000/upload_form`
2. Click "Choose File" and select your `.tif` satellite image
3. Click "Predict Water Mask"
4. View the original image and segmentation mask side-by-side


## 🔌 API Endpoints

### `GET /`
**Description**: API information and available endpoints

**Response**:
```
{
  "message": "Satellite Image Water Segmentation API",
  "version": "1.0",
  "endpoints": {
    "/predict": "POST - Upload satellite image for segmentation",
    "/health": "GET - Check API health status",
    "/upload_form": "GET - Web interface for testing"
  }
}
```

### `GET /health`
**Description**: Health check endpoint

**Response**:
```
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`
**Description**: Upload TIFF image and receive segmentation mask

**Request**:
- Content-Type: `multipart/form-data`
- Body: `image` (File) - 12-channel TIFF satellite image

**Response**:
```
{
  "success": true,
  "mask": "base64_encoded_png_image",
  "shape": ,
  "format": "base64_png"
}
```

### `POST /predict_file`
**Description**: Upload TIFF image and download mask as PNG file

**Request**: Same as `/predict`

**Response**: PNG image file (downloadable)

### `GET /upload_form`
**Description**: Web-based upload interface

**Response**: HTML page with interactive upload form

## 📦 Input Requirements

### Image Specifications
- **Format**: TIFF (.tif, .tiff)
- **Channels**: 12 bands in the following order:
  1. Coastal
  2. Blue
  3. Green
  4. Red
  5. NIR (Near-Infrared)
  6. SWIR1 (Short-Wave Infrared 1)
  7. SWIR2 (Short-Wave Infrared 2)
  8. QA Band
  9. Merit DEM (Digital Elevation Model)
  10. Copernicus DEM
  11. ESA Land Cover
  12. Water Probability

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
