import streamlit as st
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation, AutoProcessor
import requests
import io

# Load Hugging Face model and processor
model_name = 'https://huggingface.co/M-Montasser/Satellite-Segmentation-Pretrained/tree/main/model'  # Replace with your model URL
model = AutoModelForImageSegmentation.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Set up Streamlit app title and description
st.title('Satellite Image Water Body Segmentation')
st.write("This app allows you to upload a satellite image and predicts water bodies using a pre-trained segmentation model.")

# Upload Image
uploaded_file = st.file_uploader("Choose a satellite image", type=['png', 'jpg', 'jpeg', 'tiff'])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess image for the model
    inputs = processor(images=image, return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted segmentation mask
    logits = outputs.logits
    predicted_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # Display the predicted mask
    st.subheader("Predicted Water Body Mask")
    st.image(predicted_mask, caption="Predicted Mask", use_column_width=True, clamp=True)

    # Show download link for the mask image
    output_mask = Image.fromarray(predicted_mask.astype(np.uint8) * 255)
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
