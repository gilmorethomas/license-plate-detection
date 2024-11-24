import logging 
from src.pulldataset import download_dataset, parse_dataset
from src.utilities import load_image, imshow, overlay_boxes
import yaml
from ultralytics import YOLO
import os
from os import path

def train_model(model, model_config, defaults, datasetpath):
    # Return saved model if we specify we want to in defaults and it already exists in the model directory
    model_dir = model_config.get('model_directory', defaults['model_directory'])
    device = model_config.get('device', defaults['device'])

    model_name = path.join(model_dir, model_config['model_name'] + ".pt")

    if defaults['save_model'] and path.exists(model_name):
        # Load the saved model 
        return YOLO(model_name)

    logging.info(f"Training model {model_name}")
    model = YOLO(model_config['model_type'])

    # Create test / train split 
    model.data = model.data.split(0.8)
    


    model.train(data=datasetpath, # path to dataset yaml
                device=device,
                imgsz=640, epochs=model_config['epochs'])
    # Save model 
    model.save(model_name)
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    return model


def train_models(models, datasetpath):
    for modelname, model_config in models['model_metadata'].items():
        train_model(model_name=modelname, model_config=model_config, defaults=models['default_configuration'], datasetpath=datasetpath)

    # TODO Display a few of the images with the bounding boxes
    for i in range(5): 
        # Pull from train data randomly 
        ... 

        img = load_image(path.join(assumed_datasetpath, "images/Cars0.png"))
        # imshow_from_path(path.join(assumed_datasetpath, "images/Cars0.png"))
        coords = [
            # Truth data 

            # Predicted data
        ]
        coords = [
            {'x1': 226, 'y1': 125, 'x2': 419, 'y2': 173, "color": (0, 255, 0)},
            {'x1': 250, 'y1': 142, 'x2': 400, 'y2': 150, "color": (0, 0, 255)},
            ]

        img_truth = overlay_boxes(img, coords[0])
        img_predicted = overlay_boxes(img, coords[1])
        title = f"Combined Images for {path.basename(img_path).split('.')[0]""

        imshow([img, img_truth, img_predicted], "Combined Images for {img_path}", legend=["Original", "Truth", "Predicted"])

        overlay_boxes

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting execution")
    datasetpath = download_dataset() 
    data = parse_dataset(datasetpath, load_existing_annotations=False)

    # Create test and train splits


    with open("inputs/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    train_models(model_config, datasetpath)

    # models = ["yolov11n.pt"]

    # model = YOLO("yolov8m.pt")




