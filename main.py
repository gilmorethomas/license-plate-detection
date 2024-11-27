from src.logger import logger as logging 
from src.pulldataset import download_dataset, parse_dataset
from src.utilities import load_image, imshow, overlay_boxes, set_seed
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO
import shutil
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
    logging.info("Sample of training data:")
    logging.info("\n" + train_df[['xmin', 'xmax', 'ymin', 'ymax', 'img_width', 'img_height']].head().to_string(index=False))
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

    # Store the train, validation,model_co and test data in the model directory
    UpdateDataFrameToYamlFormat('train', train_df)
    UpdateDataFrameToYamlFormat('val', val_df)
    UpdateDataFrameToYamlFormat('test', test_df)

    # TODO: Create the datasets.yaml file, which could be somewhere else, and had to hard code path... don't like that.
    datasets_yaml = '''
    path: C:/Users/joeyw/CC/license-plate-detection/outputs/datasets/cars_license_plate_new

    train: train/images
    val: val/images
    test: test/images

    # number of classes
    nc: 1

    # class names
    names: ['license_plate']
    '''

    # TODO: Probably detele this line, just testing a theory
    os.chdir('C:/Users/joeyw/CC/license-plate-detection/outputs/datasets')

    # Write the content to the datasets.yaml file
    with open('datasets.yaml', 'w') as file:
        file.write(datasets_yaml)

    

    os.environ['WANDB_MODE'] = 'offline'

    # Train the model
    model_obj.train(
    data='datasets.yaml',  
    epochs=100,            
    batch=16,              
    device='cpu',         
    imgsz=320,  # Image size (width and height) for training           
    cache=True)

    return model_obj


def train_models(models, data):
    logging.info("Training models")
    for modelname, model_config in models['model_metadata'].items():
        train_model(model=modelname, model_config=model_config, defaults=models['default_configuration'], data=data)

    # Display a few of the images with the bounding boxes
    #for i in range(5): 
    #    # Pull from train data randomly 
    #    img_path = train_df.sample(1).iloc[0]['image_path']
    #    img = load_image(img_path)
    #    
    #    # Assuming the coordinates are in the dataframe
    #    coords = [
    #        {'x1': train_df.iloc[i]['xmin'], 'y1': train_df.iloc[i]['ymin'], 'x2': train_df.iloc[i]['xmax'], 'y2': train_df.iloc[i]['ymax'], "color": (0, 255, 0)},
    #        # Add predicted coordinates if available
    #    ]
    #
    #    img_truth = overlay_boxes(img, [coords[0]])
    #    # img_predicted = overlay_boxes(img, [coords[1]])  # Uncomment if predicted coordinates are available
    #    title = f"Combined Images for {path.basename(img_path).split('.')[0]}"
    #
    #    imshow([img, img_truth], title, legend=["Original", "Truth"])  # Add img_predicted if available

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




