import logging 
from src.pulldataset import download_dataset
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting execution")
    datasetpath = download_dataset() 
