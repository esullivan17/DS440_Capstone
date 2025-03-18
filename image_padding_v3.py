#this is not maintaining aspect ratio


import os
import pydicom
from PIL import Image
import numpy as np

# Define paths
folder_path = r"INSERT PATH"
save_folder = r"INSERT PATH"

# Ensure the save folder exists
os.makedirs(save_folder, exist_ok=True)

# Collect all DICOM file paths
dicom_files = []
for dirpath, _, filenames in os.walk(folder_path):
    for file in filenames:
        if file.endswith(".dcm"):
            dicom_files.append(os.path.join(dirpath, file))

# Define function: Resize images to 224 x 224 (Stretching without keeping aspect ratio)
def resize_image(image, target_size=(224, 224)):
    return image.resize(target_size, Image.BILINEAR)  # Resizing (stretching)

# Process and save images
for i, filepath in enumerate(dicom_files):
    ds = pydicom.dcmread(filepath)

    if hasattr(ds, 'pixel_array'):
        image_array = ds.pixel_array
        image = Image.fromarray((image_array / np.max(image_array) * 255).astype('uint8'))  # Convert to grayscale PIL image

        resized_image = resize_image(image)  # Resize (without keeping aspect ratio)

        # Create a filename based on the original DICOM filename
        filename = os.path.basename(filepath).replace(".dcm", ".png")
        save_path = os.path.join(save_folder, filename)

        # Save the resized image as PNG
        resized_image.save(save_path)

        print(f"Saved: {save_path}")

print("✅ All images resized and saved successfully!")
