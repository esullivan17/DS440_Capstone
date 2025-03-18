import os
import pydicom

# Define the folder containing DICOM files
folder_path = r"INSER PATH"

# Initialize list to store DICOM file paths
dicom_files = []

# Walk through all subdirectories and files
for dirpath, _, filenames in os.walk(folder_path):
    for file in filenames:
        if file.endswith(".dcm"):  # Filter for DICOM files
            dicom_files.append(os.path.join(dirpath, file))

# Debug: Print found files
print(f"Total DICOM files found: {len(dicom_files)}")

# Initialize an array for image dimensions
dim_array = []

# Read each DICOM file
for filepath in dicom_files:
    print(f"Reading file: {filepath}")

    try:
        ds = pydicom.dcmread(filepath)  # Read the DICOM file

        # Check if the file contains pixel data
        if hasattr(ds, 'pixel_array'):
            image_array = ds.pixel_array
            height, width = image_array.shape
            dim_array.append((height, width))
            print(f"Image size: {height}x{width}")
        else:
            print(f"Warning: No pixel data in {filepath}")

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

# Output all collected dimensions
print("Final dim_array:", dim_array)
