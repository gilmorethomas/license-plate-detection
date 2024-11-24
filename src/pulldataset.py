# -*- coding: utf-8 -*-

# Tommy Gilmore & Joey Wysocki
# License Plate Detection
# 
import os
from os import path  
import kagglehub
from zipfile import ZipFile
import logging
import glob 
import kagglehub
import xml.etree.ElementTree as xmlet
import pandas as pd
from src.utilities import imshow_from_path

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

def parse_dataset(datasetpath, load_existing_annotations=True):
    """Builds dataset from a dataset path. This function assumes that there is an annotations folder in the dataset path.
    This annotations folder shall contain the annotations for the dataset in an xml schema.

    Args:
        datasetpath (str, path-like): path to dataset
    """
    logging.info("Parsing dataset")
    # Unzip the dataset
    assert path.exists(datasetpath), "Dataset path does not exist"
    assert path.exists(path.join(datasetpath, "annotations")), "Dataset annotations"
    files = glob.glob(path.join(datasetpath, "annotations", "*.xml"))
    labels_dict = dict(filepath=[],imgpath=[], xmin=[],xmax=[],ymin=[],ymax=[])
    if load_existing_annotations and path.exists("data/labels.csv"):
        return pd.read_csv("data/labels.csv")
    else: 
        logging.info("Parsing dataset")
        # Parse the xml schema for the dataset
        for filename in files:

            info = xmlet.parse(filename)
            root = info.getroot()
            member_object = root.find('object')
            labels_info = member_object.find('bndbox')
            xmin = int(labels_info.find('xmin').text)
            xmax = int(labels_info.find('xmax').text)
            ymin = int(labels_info.find('ymin').text)
            ymax = int(labels_info.find('ymax').text)

            labels_dict['filepath'].append(filename)
            labels_dict['imgpath'].append(filename.replace("annotations", "images").replace("xml", "png"))
            labels_dict['xmin'].append(xmin)
            labels_dict['xmax'].append(xmax)
            labels_dict['ymin'].append(ymin)
            labels_dict['ymax'].append(ymax)

    return pd.DataFrame(labels_dict)
if __name__ == "__main__":
    datasetpath = download_dataset()
    imshow_from_path(path.join(datasetpath, "images/1.png"))