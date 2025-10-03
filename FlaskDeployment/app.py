from flask import Flask, request, jsonify, send_file
from model_utils import WaterSegmentationModel
import numpy as np
import tifffile as tiff
from PIL import Image
import io
import base64
import traceback

app = Flask(__name__)

# Load model once at startup
print("Loading model...")
model = WaterSegmentationModel(model_path='best_model.pth')
print("Model loaded successfully!")


@app.route('/')
def home():
    return jsonify({
        "message": "Satellite Image Water Segmentation API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Upload satellite image for segmentation",
            "/health": "GET - Check API health status"
        }
    })


@app.route('/upload_form')
def upload_form():
    """Simple HTML form for testing file uploads in browser"""
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Satellite Image Segmentation</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .upload-section {
                text-align: center;
                margin-bottom: 30px;
            }
            input[type="file"] {
                margin: 20px 0;
                padding: 10px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            #preview-section {
                display: none;
                margin: 20px 0;
                text-align: center;
            }
            #preview-image {
                max-width: 100%;
                max-height: 300px;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
            }
            #result {
                margin-top: 30px;
                display: none;
            }
            .comparison-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }
            .image-box {
                background: #f9f9f9;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .image-box h3 {
                margin-top: 0;
                color: #555;
            }
            .image-box img, .image-box canvas {
                max-width: 100%;
                max-height: 400px;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
            }
            #status {
                padding: 15px;
                background-color: #e3f2fd;
                border-radius: 5px;
                margin-bottom: 20px;
                text-align: center;
                font-weight: bold;
            }
            .error {
                background-color: #ffebee !important;
                color: #c62828;
            }
            .success {
                background-color: #e8f5e9 !important;
                color: #2e7d32;
            }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #4CAF50;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                vertical-align: middle;
                margin-right: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @media (max-width: 768px) {
                .comparison-container {
                    grid-template-columns: 1fr;
                }
            }
            .file-info {
                background: #e3f2fd;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 14px;
                color: #1976d2;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛰️ Satellite Water Segmentation</h1>

            <div class="upload-section">
                <p>Upload a 12-channel TIFF satellite image for water segmentation</p>

                <form id="upload-form" enctype="multipart/form-data">
                    <input type="file" name="image" id="image-input" accept=".tif,.tiff" required>
                    <br>
                    <button type="submit" id="submit-btn">Predict Water Mask</button>
                </form>
            </div>

            <div id="preview-section">
                <h3>📷 Selected File</h3>
                <div class="file-info" id="file-info"></div>
            </div>

            <div id="result">
                <div id="status"></div>

                <div class="comparison-container">
                    <div class="image-box">
                        <h3>🖼️ Original Image</h3>
                        <img id="original-image" alt="Original Image">
                    </div>

                    <div class="image-box">
                        <h3>💧 Water Segmentation Mask</h3>
                        <img id="mask-image" alt="Segmentation Mask">
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/tiff.js@latest/tiff.min.js"></script>
        <script>
            const fileInput = document.getElementById('image-input');
            const previewSection = document.getElementById('preview-section');
            const form = document.getElementById('upload-form');
            const submitBtn = document.getElementById('submit-btn');

            let selectedFile = null;
            let originalImageDataUrl = null;

            // Preview file info when selected
            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    selectedFile = file;
                    const fileInfo = document.getElementById('file-info');
                    fileInfo.innerHTML = `
                        <strong>✓ File Selected:</strong> ${file.name}<br>
                        <strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB
                    `;
                    previewSection.style.display = 'block';
                }
            });

            // Handle form submission
            form.onsubmit = async (e) => {
                e.preventDefault();

                if (!selectedFile) {
                    alert('Please select a file first');
                    return;
                }

                const formData = new FormData();
                formData.append('image', selectedFile);

                const resultDiv = document.getElementById('result');
                const statusDiv = document.getElementById('status');

                // Show loading state
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="loading"></span>Processing...';
                statusDiv.innerHTML = '<span class="loading"></span>Processing image...';
                statusDiv.className = '';
                resultDiv.style.display = 'block';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (data.success) {
                        // Show success message
                        statusDiv.textContent = `✅ Success! Mask shape: ${data.shape[0]} x ${data.shape[1]}`;
                        statusDiv.className = 'success';

                        // Get preview of original TIFF (convert to viewable format)
                        await displayOriginalImage(selectedFile);

                        // Display segmentation mask
                        const maskImg = document.getElementById('mask-image');
                        maskImg.src = 'data:image/png;base64,' + data.mask;
                        maskImg.style.display = 'block';

                    } else {
                        statusDiv.textContent = '❌ Error: ' + (data.error || 'Unknown error');
                        statusDiv.className = 'error';
                        document.getElementById('mask-image').style.display = 'none';
                        document.getElementById('original-image').style.display = 'none';
                    }
                } catch (error) {
                    statusDiv.textContent = '❌ Error: ' + error.message;
                    statusDiv.className = 'error';
                    document.getElementById('mask-image').style.display = 'none';
                    document.getElementById('original-image').style.display = 'none';
                } finally {
                    // Reset button state
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Predict Water Mask';
                }
            };

            // Function to display original TIFF image
            async function displayOriginalImage(file) {
                const reader = new FileReader();

                return new Promise((resolve, reject) => {
                    reader.onload = function(e) {
                        try {
                            const buffer = e.target.result;

                            // Parse TIFF using tiff.js library
                            Tiff.initialize({TOTAL_MEMORY: 16777216 * 10});
                            const tiff = new Tiff({buffer: buffer});
                            const canvas = tiff.toCanvas();

                            if (canvas) {
                                // Convert canvas to data URL
                                const dataUrl = canvas.toDataURL();
                                const originalImg = document.getElementById('original-image');
                                originalImg.src = dataUrl;
                                originalImg.style.display = 'block';
                                resolve();
                            } else {
                                // Fallback if TIFF can't be rendered
                                showPlaceholder();
                                resolve();
                            }
                        } catch (error) {
                            console.error('Error displaying TIFF:', error);
                            showPlaceholder();
                            resolve();
                        }
                    };

                    reader.onerror = function() {
                        showPlaceholder();
                        resolve();
                    };

                    reader.readAsArrayBuffer(file);
                });
            }

            // Fallback placeholder if TIFF can't be displayed
            function showPlaceholder() {
                const originalImg = document.getElementById('original-image');
                originalImg.style.display = 'none';
                const parent = originalImg.parentElement;
                parent.innerHTML += '<div style="padding: 40px; background: #f0f0f0; border-radius: 5px; margin-top: 10px;">📄 TIFF file (preview not available)</div>';
            }
        </script>
    </body>
    </html>
    '''


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": True})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict water segmentation mask
    Expected input: multipart/form-data with 'image' field containing TIFF file
    Returns: JSON with base64 encoded mask image
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Read image based on file type
        if file.filename.lower().endswith(('.tif', '.tiff')):
            # Read TIFF file
            image_bytes = file.read()
            image_array = tiff.imread(io.BytesIO(image_bytes)).astype(np.float32)
        else:
            return jsonify({"error": "Only TIFF files are supported"}), 400

        # Validate image shape (should have 12 channels)
        if image_array.ndim != 3 or image_array.shape[-1] != 12:
            return jsonify({
                "error": f"Invalid image shape. Expected (H, W, 12), got {image_array.shape}"
            }), 400

        # Make prediction
        mask = model.predict(image_array)

        # Convert mask to PNG image
        mask_image = Image.fromarray(mask * 255, mode='L')

        # Convert to base64 for JSON response
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        buffer.seek(0)
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            "success": True,
            "mask": mask_base64,
            "shape": mask.shape,
            "format": "base64_png"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/predict_file', methods=['POST'])
def predict_file():
    """
    Alternative endpoint that returns mask as downloadable PNG file
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']

        # Read TIFF
        image_bytes = file.read()
        image_array = tiff.imread(io.BytesIO(image_bytes)).astype(np.float32)

        # Predict
        mask = model.predict(image_array)

        # Convert to image
        mask_image = Image.fromarray(mask * 255, mode='L')

        # Save to buffer
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png', as_attachment=True,
                         download_name='segmentation_mask.png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
