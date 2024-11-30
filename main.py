import comet_ml

from src.logger import logger as logging 
from src.pulldataset import download_dataset, parse_dataset
from src.utilities import load_image, imshow, overlay_boxes, set_seed, verify_device, start_resource_utilization_thread, stop_resource_utilization_thread, plot_resource_utilization, convert_xy_bounds_to_centered_xywh

from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO
import shutil
import os
from os import path, makedirs
from src.plotting import pie_chart
import pandas as pd 
from src.globals import LICENSE_PLATE_GLOBALS as LPG 
import time
from inspect import signature, Parameter

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
    model_output_dir = path.join(LPG.OUTPUTS_DIR, model)

    train_split = model_config.get('data_split', defaults['data_split']).get('train_split', defaults['data_split']['train_split'])
    test_split = model_config.get('data_split', defaults['data_split']).get('test_split', defaults['data_split']['test_split'])
    validation_split = model_config.get('data_split', defaults['data_split']).get('validation_split', defaults['data_split']['validation_split'])
    # Get correct device, verifying that the selected device exists. If not, then run with the cpu 
    device = verify_device(model_config['training_parameters'].get('device', defaults['training_parameters']['device']))
    # model_name = path.join(LPG.OUTPUTS_DIR, model_dir, + ".pt")
    # # If the model ends in a number, we want to append an underscore to the model name
    # if model_output[-1].isdigit():
    #     model_name += "_"
    #     logging.info(f"Model name ends in a number. Appending underscore to model name: {model_name}")

    # We need to look for the latest directory in the model directory matching the model name. This is denoted by the highest number at the end of the directory name. Assume that we are going to start at 0 
    # model_dir = path.join(LPG.OUTPUTS_DIR, model_dir)
    if path.exists(model_output_dir):
        # Get the latest model directory
        # If there are no directories, then we start at 0
        if not os.listdir(model_output_dir):
            latest_model_dir = 0
        else:
            latest_model_dir = max([int(d) for d in os.listdir(model_output_dir) if os.path.isdir(os.path.join(model_output_dir, d))]) + 1
    else:
        latest_model_dir = 0
    # Do not make the directory, since the yolo object will do that for us    
    model_dir = path.join(model_output_dir, str(latest_model_dir))
    model_name = path.join(model_dir, model + ".pt")

    if model_config.get('load_saved_models', defaults['load_saved_model']):
        expected_dir = path.join(model_output_dir, str(latest_model_dir - 1))
        if path.exists(expected_dir) and path.exists(path.join(expected_dir, model + '.pt')) and path.exists(path.join(expected_dir, 'weights', 'best.onnx')):
            # Load the saved model from the model directory .pt file 
            logging.info(f"Loading model {model_name}")
            saved_model = YOLO(path.join(expected_dir, model + '.pt'))
            best_model = YOLO(path.join(expected_dir, 'weights', 'best' + '.pt'))
            return {'model_obj' : saved_model, 'output_dir': expected_dir, 'train_results': None, 'best_model': best_model}

            # Load the best model from the weights directory
            
            # Load the best weights from the weights directory
            # ret_model.load_weights(path.join(expected_dir, 'weights', 'best.onnx'))

    elif model_config.get('load_saved_models', defaults['load_saved_model']):
        logging.warning(f"You specified to load saved model, but model {model_name} does not exist. Training new model.")
        

    logging.info(f"Training model {model_name}")
    model_obj = YOLO(model_config['model_type'])

    # Build test / train split 
    # Randomly pre-filter the data based on the train, validation, and test split. If the sum of the splits is less than 1, the remaining data will be unused.
    if train_split + test_split + validation_split > 1:
        raise ValueError("Sum of train, test, and validation splits must be less than or equal to 1")
    elif train_split + test_split + validation_split == 1:
        # Shuffle data 
        new_data = data.sample(frac=1, random_state=model_config.get('seed', defaults['seed']))
        # Make remaining data of size 0 with same columns as new_data
        remaining_points = pd.DataFrame(columns=new_data.columns)
    else: 
        new_data, remaining_points = train_test_split(data, test_size=1 - (train_split + test_split + validation_split), random_state=model_config.get('seed', defaults['seed']))
    # Since the split is based on the original size of the data, we need to scale the split sizes based on the new data size
    adj_train_split = train_split / (train_split + test_split + validation_split)
    adj_test_split = test_split / (test_split + validation_split)
    adj_validation_split = validation_split / (test_split + validation_split)
    train_df, tmp = train_test_split(new_data, test_size= 1 - adj_train_split, random_state=model_config.get('seed', defaults['seed']))
    logging.info("Sample of training data:")
    logging.info("\n" + train_df[['xmin', 'xmax', 'ymin', 'ymax', 'img_width', 'img_height']].head().to_string(index=False))
    val_df, test_df = train_test_split(tmp, test_size= 1 - adj_validation_split, random_state=model_config.get('seed', defaults['seed']))
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Create a pie chart using plotly for the sizes of the train, validation, and test df sizes  
    if verbose:
        logging.info(f"Train size: {len(train_df)}")
        logging.info(f"Validation size: {len(val_df)}")
        logging.info(f"Test size: {len(test_df)}")
        logging.info(f"Outputting dataset split to {model_dir}")
        logging.info(f"Remaining (unused) data size: {len(remaining_points)}")
        pie_chart(data=pd.DataFrame({'size': [len(train_df), len(val_df), len(test_df), len(remaining_points)], 'type': ['Train Split', 'Validation Split', 'Test Split', 'Remaining Data']}),
                  values='size', names='type', title='Dataset Split', filename='dataset_split', output_dir=path.join(LPG.PLOTS_DIR, model))

    # Store the train, validation,model_co and test data in the model directory

    new_data_dir = path.join(LPG.OUTPUTS_DIR, 'datasets', 'cars_license_plate_new')

    interface_yaml = createYamlFormattedData(train_df, val_df, test_df, new_data_dir)
    

    # os.environ['WANDB_MODE'] = 'offline'

    # Train the model
    # Load train arguments from the default model configuration. 
    train_args = defaults['training_parameters']
    # Override each train kwarg in the defaults with one from the model's train_args if they exist.
    for key, value in model_config.get('training_parameters', {}).items():
        train_args[key] = value

    # If device is in our dict, get rid of it 
    if 'device' in train_args:
        train_args.pop('device')
    train_results = model_obj.train(
        data=interface_yaml,  
        project=f'outputs/{model}', # Set the output directory for the model
        # project=f'outputs/{model}/runs', # Set the output directory for the model
        # project=LPG.PROJECT_NAME,  # Comet project name
        name = latest_model_dir,  # Name of the experiment
        device=device,  # Device to train on
        seed=model_config.get('seed', defaults['seed']),  # Seed for reproducibility
        **train_args # Additional training arguments
    ,)

    logging.info(f"Saving model object to {model_name}")
    makedirs(path.dirname(model_name), exist_ok=True)
    try: 
        model_obj.save(model_name)
    except Exception as e:
        logging.error(f"Failed to save model {model_name}: {e}")
    # Export the model to ONNX format
    try:
        outpath = model_obj.export(format="onnx")  # return path to exported model
        logging.info(f"Exported model to ONNX format: {outpath}")
    except Exception as e:
        import pdb; pdb.set_trace()
        logging.error(f"Failed to export model to ONNX format: {e}")
        outpath = None
    return {'model_obj' : model_obj, 'train_results': train_results, 'output_dir': model_dir, 'best_model': model_obj}


def train_models(models, data):
    """Trains all models 

    Args:
        models (dict): Model configuration
        data (pd.DataFrame): Data to train on. Contains image paths and bounding box coordinates

    Returns:
        dict: Dictionary of trained models
    """
    logging.info("Training models")
    model_objs = {}
    for modelname, model_config in models['model_metadata'].items():
        model_objs[modelname] = train_model(model=modelname, model_config=model_config, defaults=models['default_configuration'], data=data)
    
    return model_objs

def validate_model(model, data):
    """Validate a model
    """
    logging.info(f"Validating model {model}")
    results = model.val()

    ...
def validate_models(models, data):
    ...
    for modelname, model_obj in models.items():
        validate_model(modelname, model_obj)
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

def test_model(model, data):
    ...

def createYamlFormattedData(train_df, val_df, test_df, output_dir):
    """Creates the YOLO format for the data and saves the labels and images to the appropriate directories

    Args:
        train_df (pd.DataFrame): Training data containing image paths and bounding box coordinates
        val_df (pd.DataFrame): Validation data containing image paths and bounding box coordinates
        test_df (pd.DataFrame): Testing dataframe
        output_dir (str, path-like): Output directory
    
    Returns:
        str: Path to the datasets.yaml file
    """
    updateDataFrameToYamlFormat('train', train_df, output_dir=output_dir)
    updateDataFrameToYamlFormat('val', val_df, output_dir=output_dir)
    updateDataFrameToYamlFormat('test', test_df, output_dir=output_dir)

    datasets_yaml = f'''
    path: {output_dir}

    train: train/images
    val: val/images
    test: test/images

    # number of classes
    nc: 1

    # class names
    names: ['license_plate']
    '''

    # Write the content to the datasets.yaml file
    output_file_name = path.join(output_dir, 'datasets.yaml')
    with open(output_file_name, 'w') as file:
        logging.info(f"Writing datasets.yaml to {path.join(output_dir, 'datasets.yaml')}")
        file.write(datasets_yaml)
    
    return output_file_name
    
def updateDataFrameToYamlFormat(split_name, df, output_dir):
    """Converts a DataFrame to the YOLO format and saves the labels and images to the appropriate directories

    Args:
        split_name (str): Name of the split (train, test, validation)
        output_dir (pd.DataFrame): DataFrame containing image paths and bounding box coordinates
        output_dir (str): Output directory to save the labels and images
    """
    # Define paths for labels and images
    labels_path = os.path.join(output_dir, split_name, 'labels') 
    images_path = os.path.join(output_dir, split_name, 'images')

    # Create directories if they don't exist
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        img_name = row['imgname']
        img_extension = '.png'

        # Calculate YOLO format coordinates, which are normalized and based on box center and width/height
        row_df = convert_xy_bounds_to_centered_xywh(row)
        x_center = row_df['x_center']
        y_center = row_df['y_center']
        width = row_df['width']
        height = row_df['height']

        # Save labels in YOLO format
        label_path = os.path.join(labels_path, f'{img_name}.txt')
        with open(label_path, 'w') as file:
            file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

        # Copy image to the images directory
        try:
            shutil.copy(row['imgpath'], os.path.join(images_path, img_name + img_extension))
        except Exception as e:
            logging.error(f"Failed to copy image {row['imgpath']} to {os.path.join(images_path, img_name + img_extension)}: {e}")

    logging.info(f"Created '{images_path}' and '{labels_path}'")


def configure_comet_ml():
    """Configure Comet ML
    """
    # Configure Comet ML
    # from comet_ml import Experiment
    # experiment = Experiment
    logging.warning("Attempting to configure Comet ML. You must have a Comet account and API key to use this feature.")

    comet_ml.login(project_name="license-plate-detection")
    # Set the comet config 
    os.environ['COMET_CONFIG'] = LPG.COMET_CONFIG

    # Use if running offline experiment. You can later upload the directory 
    # os.environ["COMET_MODE"] = "offline"


if __name__ == '__main__':
    logging.setLevel("INFO")

    # logging.basicConfig(level=logging.INFO)
    # logging.info("Starting execution")
    # Start thread for resource utilization
    stop_event, logging_thread, cpu_usage, memory_usage, gpu_usage, timestamps = start_resource_utilization_thread()
    if LPG.USE_COMET:
        configure_comet_ml()
    datasetpath = download_dataset() 
    data = parse_dataset(datasetpath, load_existing_annotations=False)


    # Create test and train splits


    with open("inputs/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    train_models(model_config, data)

    stop_resource_utilization_thread(stop_event, logging_thread)
    plot_resource_utilization(cpu_usage, memory_usage, gpu_usage, timestamps, LPG.PLOTS_DIR)