import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image
import tifffile as tiff


class WaterSegmentationModel:
    def __init__(self, model_path='best_model.pth', device=None):
        """Initialize the segmentation model"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model architecture
        self.model = smp.Unet(
            encoder_name='mobilenet_v2',
            encoder_weights=None,  # Don't load pretrained weights
            in_channels=11,
            classes=1,
            activation=None
        )

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image_array):
        """
        Preprocess satellite image: normalize, generate features, remove channels
        Args:
            image_array: numpy array of shape (H, W, 12) - original 12 channels
        Returns:
            processed image ready for model input
        """
        # Normalize each channel
        normalized_image = np.empty_like(image_array, dtype=np.float32)
        for ch in range(image_array.shape[-1]):
            ch_data = image_array[:, :, ch]
            ch_min, ch_max = np.min(ch_data), np.max(ch_data)
            normalized_image[:, :, ch] = (ch_data - ch_min) / (ch_max - ch_min + 1e-9)

        # Extract channels for feature generation
        red = normalized_image[:, :, 3]
        nir = normalized_image[:, :, 4]
        green = normalized_image[:, :, 2]

        # Generate NDWI and NDVI
        ndwi = (green - nir) / (green + nir + 1e-9)
        ndvi = (nir - red) / (nir + red + 1e-9)

        # Add new channels
        ndwi = np.expand_dims(ndwi, axis=-1)
        ndvi = np.expand_dims(ndvi, axis=-1)
        featured_image = np.concatenate([normalized_image, ndwi, ndvi], axis=-1)

        # Remove channels [1, 2, 3] (Blue, Green, Red)
        channels_to_remove = [1, 2, 3]
        featured_image = np.delete(featured_image, obj=channels_to_remove, axis=-1)

        return featured_image

    def predict(self, image_array):
        """
        Make prediction on preprocessed image
        Args:
            image_array: numpy array of shape (H, W, 12)
        Returns:
            binary mask as numpy array
        """
        # Preprocess
        processed = self.preprocess_image(image_array)

        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=torch.float)

        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.sigmoid(output) > 0.5

        # Convert back to numpy
        mask = prediction.squeeze().cpu().numpy().astype(np.uint8)
        return mask
