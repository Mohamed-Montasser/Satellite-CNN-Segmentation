import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import tifffile as tiff  # Import tifffile
from PIL import Image
from transformers import AutoModelForImageSegmentation, AutoProcessor

# Define your model architecture (Assuming a U-Net model here)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(11, 64, kernel_size=3, padding=1),  # Assuming 11 input channels
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # 1 output channel for binary segmentation
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load model from .pth file
def load_model(model_path):
    model = UNet()  # Initialize your model
    model.load_state_dict(torch.load(model_path))  # Load the model weights
    model.eval()  # Set the model to evaluation mode
    return model

# Set up Streamlit app title and description
st.title('Satellite Image Water Body Segmentation')
st.write("This app allows you to upload a satellite image in TIFF format and predicts water bodies using a pre-trained segmentation model.")

# Upload Image
uploaded_file = st.file_uploader("Choose a satellite image (TIFF format)", type=['tiff'])

if uploaded_file is not None:
    # Open the TIFF image using tifffile
    image = tiff.imread(uploaded_file)

    # Check the shape of the image
    st.write(f"Image Shape: {image.shape}")

    # Display only the first channel (for example, the first band or RGB-like band)
    st.image(image[:,:,0], caption='First Band', use_column_width=True, clamp=True)  # Adjust if needed

    # Preprocess the image for the model (convert to tensor and normalize)
    image = np.array(image).astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Change to (1, C, H, W)
    
    # Normalize the image if needed (example normalization)
    image = image / 255.0  # Assuming input is in the range [0, 255]

    # Load model from the .pth file
    model_path = 'M-Montasser/Satellite-Segmentation-Pretrained'  # Specify the path to your .pth file
    model = AutoModelForImageSegmentation.from_pretrained(model_path)
    # Make prediction
    with torch.no_grad():
        output = model(image)  # Forward pass

    # Convert the output to a mask
    predicted_mask = output.squeeze().cpu().numpy()
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Threshold for binary segmentation

    # Display the predicted mask
    st.subheader("Predicted Water Body Mask")
    st.image(predicted_mask, caption="Predicted Mask", use_column_width=True, clamp=True)

    # Show download link for the mask image
    output_mask = Image.fromarray(predicted_mask * 255)  # Convert back to [0, 255]
    output_mask_path = '/tmp/output_mask.png'
    output_mask.save(output_mask_path)
    
    with open(output_mask_path, 'rb') as file:
        btn = st.download_button(
            label="Download Predicted Mask",
            data=file,
            file_name="predicted_mask.png",
            mime="image/png"
        )

else:
    st.write("Upload an image to start.")
