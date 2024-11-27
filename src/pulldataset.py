# -*- coding: utf-8 -*-

# Tommy Gilmore & Joey Wysocki
# License Plate Detection
# 
import os
from os import path  
import kagglehub
from zipfile import ZipFile
import glob 
import kagglehub
import xml.etree.ElementTree as xmlet
import pandas as pd
from src.utilities import imshow_from_path
from src.logger import logger as logging 
import shutil

# Download latest version

def download_dataset():
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
    labels_dict = dict(filepath=[],imgpath=[], imgname = [], xmin=[],xmax=[],ymin=[],ymax=[], img_width=[], img_height=[])
    if load_existing_annotations and path.exists("data/labels.csv"):
        return pd.read_csv("data/labels.csv")
    else: 
        logging.info("Parsing dataset")
        # Parse the xml schema for the dataset
        for filename in files:

            info = xmlet.parse(filename)
            root = info.getroot()
            member_name   = root.find('filename')
            member_object = root.find('object')
            member_size   = root.find('size')
            labels_info   = member_object.find('bndbox')
            xmin = int(labels_info.find('xmin').text)
            xmax = int(labels_info.find('xmax').text)
            ymin = int(labels_info.find('ymin').text)
            ymax = int(labels_info.find('ymax').text)
            img_width = int(member_size.find('width').text)
            img_height = int(member_size.find('height').text)
            # Pull out image name, remove the .png extension
            imgname    = member_name.text.split(".")[0]

            labels_dict['filepath'].append(filename)
            labels_dict['imgpath'].append(filename.replace("annotations", "images").replace("xml", "png"))
            labels_dict['imgname'].append(imgname)
            labels_dict['xmin'].append(xmin)
            labels_dict['xmax'].append(xmax)
            labels_dict['ymin'].append(ymin)
            labels_dict['ymax'].append(ymax)
            labels_dict['img_width'].append(img_width)  
            labels_dict['img_height'].append(img_height)    

    return pd.DataFrame(labels_dict)


def UpdateDataFrameToYamlFormat(split_name, Input_Dataframe):
    # Define paths for labels and images
    labels_path = os.path.join(LPG.OUTPUTS_DIR, 'datasets', 'cars_license_plate_new', split_name, 'labels')
    images_path = os.path.join(LPG.OUTPUTS_DIR, 'datasets', 'cars_license_plate_new', split_name, 'images')

    # Create directories if they don't exist
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    # Iterate over each row in the DataFrame
    for _, row in Input_Dataframe.iterrows():
        img_name = row['imgname'];
        img_extension = '.png'

        # Calculate YOLO format coordinates
        x_center = (row['xmin'] + row['xmax']) / 2 / row['img_width']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['img_height']
        width = (row['xmax'] - row['xmin']) / row['img_width']
        height = (row['ymax'] - row['ymin']) / row['img_height']

        # Save labels in YOLO format
        label_path = os.path.join(labels_path, f'{img_name}.txt')
        with open(label_path, 'w') as file:
            file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

        # Copy image to the images directory
        try:
            shutil.copy(row['imgpath'], os.path.join(images_path, img_name + img_extension))
        except Exception as e:
            logging.error(f"Failed to copy image {row['imgpath']} to {os.path.join(images_path, img_name + img_extension)}: {e}")

    print(f"Created '{images_path}' and '{labels_path}'")
if __name__ == "__main__":
    datasetpath = download_dataset()
    imshow_from_path(path.join(datasetpath, "images/1.png"))