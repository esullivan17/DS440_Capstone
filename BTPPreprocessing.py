# %%
#Code to load in data files
import pydicom
import os
from PIL import Image
from torchvision import transforms

def Load_DCM_Files(base_path):
    DCM_Files = []

    for root, dirs, files in os.walk(base_path):
        for file in files: 
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                try: 
                    DCM_Data = pydicom.dcmread(file_path)
                    DCM_Files.append(DCM_Data)
                    print(f"Loaded DCM File: {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return DCM_Files

Directory = r"C:\Users\hgood\OneDrive\Documents\Brain-Tumor-Progression"
DCM_Files = Load_DCM_Files(Directory)




# %% [markdown]
# Datasets are loaded, now we need to make sure that all images are sized equally, and then convert the images to tensors to apply to our neural nets.

# %%
#Code to load in data files
import pydicom
import os
from PIL import Image
from torchvision import transforms

def Load_DCM_Files(base_path):
    DCM_Files = []

    #TransformImages will resize, normalize, and transform the images to tensors to use in our CNN.
    TransformImages = transforms.Compose([
        transforms.Resize((224,224)), #Resizing all images to 224 x 224
        transforms.ToTensor(),#Converting Images to Tensors
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])#Normalizations
    ])

    for root, dirs, files in os.walk(base_path):
        for file in files: 
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                try: 
                    DCMData = pydicom.dcmread(file_path)
                    DCMImages = Image.fromarray(DCMData.pixel_array)
                    DCMTransformed = TransformImages(DCMImages) 
                    DCM_Files.append(DCMTransformed)
                    print(f"Loaded DCM File: {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return DCM_Files

Directory = r"C:\Users\hgood\OneDrive\Documents\Brain-Tumor-Progression"
DCM_Files = Load_DCM_Files(Directory)



