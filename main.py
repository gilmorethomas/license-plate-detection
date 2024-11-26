from src.logger import logger as logging 
from src.pulldataset import download_dataset, parse_dataset
from src.utilities import load_image, imshow, overlay_boxes, set_seed
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO
import os
from os import path
from src.plotting import pie_chart
import pandas as pd 
from src.globals import LICENSE_PLATE_GLOBALS as LPG 

def train_model(model, model_config, defaults, data, verbose=True):
    """Train a model
    Utilizes the model configuration to set up the model and train it on the data.

    Args:
        model (str): Model name
        model_config (dict): Model configuration
        defaults (dict): Default configuration 
        data (pd.DataFrame): Data to train on. Contains image paths and bounding box coordinates

    Returns:
        _type_: _description_
    """
    # Set seed
    logging.info(f"Training model {model}")
    set_seed(model_config.get('seed', defaults['seed']))

    # Return saved model if we specify we want to in defaults and it already exists in the model directory
    model_dir = model_config.get('model_directory', defaults['model_directory'])
    device = model_config.get('device', defaults['device'])
    train_split = model_config.get('train_split', defaults['train_split'])
    test_split = model_config.get('test_split', defaults['test_split'])
    validation_split = model_config.get('validation_split', defaults['validation_split'])
    model_name = path.join(LPG.MODEL_DIR, model + ".pt")

    if defaults['load_saved_model'] and path.exists(model_name):
        # Load the saved model from the model directory .pt file 
        logging.info(f"Loading model {model_name}")
        return YOLO(model_name)

    logging.info(f"Training model {model_name}")
    model_obj = YOLO(model_config['model_type'])

    # Build test / train split 
    train_df, tmp = train_test_split(data, test_size= 1 - train_split, random_state=model_config.get('seed', defaults['seed']))
    val_df, test_df = train_test_split(tmp, test_size=1 - (validation_split / (test_split + validation_split)), random_state=model_config.get('seed', defaults['seed']))
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Create a pie chart using plotly for the sizes of the train, validation, and test df sizes  
    if verbose:
        logging.info(f"Train size: {len(train_df)}")
        logging.info(f"Validation size: {len(val_df)}")
        logging.info(f"Test size: {len(test_df)}")
        logging.info(f"Outputting dataset split to {model_dir}")
        pie_chart(data=pd.DataFrame({'size': [len(train_df), len(val_df), len(test_df)], 'type': ['Train Split', 'Validation Split', 'Test Split']}),
                  values='size', names='type', title='Dataset Split', filename='dataset_split', output_dir=path.join(LPG.PLOTS_DIR, model))






    # # Create test / train split 
    # model.data = model.data.split(train_split)

    # model.train(data=data, # path to dataset yaml
    #             device=device,
    #             imgsz=640, epochs=model_config['epochs'])
    # # Save model 
    # model.save(model_name)
    # if not path.exists(model_dir):
    #     os.makedirs(model_dir)
    return model_obj


def train_models(models, data):
    logging.info("Training models")
    for modelname, model_config in models['model_metadata'].items():
        train_model(model=modelname, model_config=model_config, defaults=models['default_configuration'], data=data)

    # # TODO Display a few of the images with the bounding boxes
    # for i in range(5): 
    #     # Pull from train data randomly 
    #     ... 

    #     img = load_image(path.join(assumed_datasetpath, "images/Cars0.png"))
    #     # imshow_from_path(path.join(assumed_datasetpath, "images/Cars0.png"))
    #     coords = [
    #         # Truth data 

    #         # Predicted data
    #     ]
    #     coords = [
    #         {'x1': 226, 'y1': 125, 'x2': 419, 'y2': 173, "color": (0, 255, 0)},
    #         {'x1': 250, 'y1': 142, 'x2': 400, 'y2': 150, "color": (0, 0, 255)},
    #         ]

    #     img_truth = overlay_boxes(img, coords[0])
    #     img_predicted = overlay_boxes(img, coords[1])
    #     title = f"Combined Images for {path.basename(img_path).split('.')[0]""

    #     imshow([img, img_truth, img_predicted], "Combined Images for {img_path}", legend=["Original", "Truth", "Predicted"])

    #     overlay_boxes

if __name__ == '__main__':
    logging.setLevel("INFO")

    # logging.basicConfig(level=logging.INFO)
    # logging.info("Starting execution")
    datasetpath = download_dataset() 
    data = parse_dataset(datasetpath, load_existing_annotations=False)

    # Create test and train splits


    with open("inputs/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    train_models(model_config, data)

    # models = ["yolov11n.pt"]

    # model = YOLO("yolov8m.pt")




