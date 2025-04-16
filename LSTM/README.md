# REMBRANDT MRI Preprocessing Pipeline

This repository contains multiple versions of preprocessing scripts used to convert REMBRANDT brain tumor MRI datasets into structured tensor inputs for LSTM-based classification models.

## Overview
Each script processes DICOM image slices into tensors using a combination of:
- Rescaling (using `RescaleSlope` and `RescaleIntercept`)
- Normalization to [0, 1]
- Resizing to 256×256
- Wavelet transforms (DWT using Haar basis)
- Discrete Cosine Transform (DCT)
- Optional KMeans clustering for tumor region emphasis

The outputs are dictionaries serialized with PyTorch, mapping each patient ID to their tumor grade and list(s) of extracted feature tensors.

---

##Recommended script as of 4/16/2025: 

## Preprocessing Scripts

### `REMBRANDT_Processing_byFolder_v2`
**Steps:** Rescaling, normalization, resizing, clustering (k=3), masking, DWT, DCT  
**Input CSV:** `clinical_cleaned_v2.csv`  
**Output Structure:**
```python
{
  "900-00-1961": {
    "grade": 2,
    "sequences": [
      [tensor(100), tensor(100)],  # Folder A
      [tensor(100), tensor(100)],  # Folder B
    ]
  }
}
```
**Output File:** `patient_feature_data_v6.pt`

### `REMBRANDT_Processing_byFolder_v1`
**Steps:** Rescaling, normalization, resizing, DWT, DCT (no clustering)  
**Input CSV:** `clinical_cleaned_v2.csv`  
**Output File:** `patient_feature_data_v5.pt`

### `REMBRANDT_Processing_subset_v3`
**Steps:** Rescaling, normalization, resizing, DWT, DCT  
**Input:** Hardcoded 9 patients  
**Output Structure:**
```python
{
  "900-00-5468": {
    "grade": 2,
    "slices": [
      {"features": tensor(100), "rel_path": "..."},
      ...
    ]
  }
}
```
**Output File:** `patient_feature_data_v3.pt`

### `REMBRANDT_Processing_subset_v4`
**Changes from v3:** Normalization removed (to preserve contrast)  
**Output File:** `patient_feature_data_v4.pt`

### `REMBRANDT_Processing_subset_v5`
**Changes from v4:** Removed `rel_path` field  
**Output File:** `patient_feature_data_v5.pt`

### `REMBRANDT_Processing_full_bare`
**Steps:** Rescaling, normalization, resizing (no wavelet or DCT)  
**Input CSV:** `clinical_cleaned_v1.csv`  
**Output Structure:** Raw 256×256 image tensors per slice  
**Output File:** `patient_feature_data_bare.pt` (assumed)

### `REMBRANDT_Processing_full_NC`
**Steps:** Rescaling, normalization, resizing, DWT, DCT (no clustering)  
**Input CSV:** `clinical_cleaned_v1.csv`  
**Output File:** Not explicitly defined

---

## Usage
Each script can be executed directly in Python. All required dependencies include:
- `pydicom`
- `numpy`
- `PIL`
- `torch`
- `pywt`
- `scipy`
- `pandas`
- `sklearn`

Processed files (e.g., `patient_feature_data_v6.pt`) are saved using `torch.save()` and are compatible with PyTorch LSTM-based classifiers.

---

## Author
Developed by Ethan Sullivan as part of a capstone research project for brain tumor grade classification using temporal modeling of MRI sequences.

---

## License
MIT License
