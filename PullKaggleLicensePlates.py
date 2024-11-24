# -*- coding: utf-8 -*-

# Tommy Gilmore & Joey Wysocki
# License Plate Detection
# 
# Need to run: pip install kaggle & pip install ultralytics
import os
from zipfile import ZipFile


# Set Kaggle API key location
os.environ['KAGGLE_CONFIG_DIR'] = '/path/to/.kaggle/'  # Replace with the actual path to your `kaggle.json`

# Define the dataset path
dataset_path = './data/car-plate-detection.zip'
extracted_path = './data/car-plate-detection'

# Download the dataset
print("Downloading dataset from Kaggle...")
os.system('kaggle datasets download -d andrewmvd/car-plate-detection -p ./data')
print(f"Dataset downloaded and saved at: {dataset_path}")

# Check if the dataset was downloaded
if os.path.exists(dataset_path):
    print(f"Dataset successfully downloaded. File size: {os.path.getsize(dataset_path) / 1024 / 1024:.2f} MB")
else:
    print("Dataset download failed. Please check your Kaggle API key or internet connection.")
    exit()

# Unzip the dataset
print("Extracting dataset...")
with ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)
print(f"Dataset extracted to: {extracted_path}")

# List extracted files
print("Extracted files and directories:")



