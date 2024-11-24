# -*- coding: utf-8 -*-

# Tommy Gilmore & Joey Wysocki
# License Plate Detection
# 
import os
from os import path  
import kagglehub
from zipfile import ZipFile
import logging

import kagglehub

# Download latest version

def download_dataset():
    logging.basicConfig(level=logging.INFO)
    assumed_datasetpath = path.join(os.path.expanduser("~"), ".cache/kagglehub/datasets/andrewmvd/car-plate-detection/versions/1")

    if path.exists(assumed_datasetpath): 
        logging.info(f"Dataset already exists. Skipping download.")
    else:
        logging.info("Downloading dataset")
        dataset_path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
        if path.exists(dataset_path):
            logging.info(f"Dataset downloaded to {dataset_path}")
        else:
            logging.error("Dataset download failed. Please check your Kaggle API key or internet connection.")
            exit()
        assumed_datasetpath = dataset_path
    return assumed_datasetpath

if __name__ == "__main__":
    download_dataset()