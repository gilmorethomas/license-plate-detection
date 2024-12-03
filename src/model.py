from sklearn.model_selection import train_test_split
from ultralytics import YOLO

from src.logger import logger as logging 
from src.plotting import pie_chart
import pandas as pd 
import os
from os import path, makedirs
from src.globals import LICENSE_PLATE_GLOBALS as LPG 
from src.utilities import load_image, imshow, overlay_boxes, set_seed, verify_device, start_resource_utilization_thread, stop_resource_utilization_thread, plot_resource_utilization, convert_xy_bounds_to_centered_xywh
from src.pulldataset import createYamlFormattedData

def test_models(models, model_configs, output_dir):
    """Gets the test results for all models, which is comprised of detections on the test data

    Args:
        models (dict): Dictionary of model names to model objects and metadata
        model_configs (dict): Model configuration, loaded from yaml 

    Returns:
        models: Dictionary of model names to model objects updated with test results
    """
    logging.info("Testing models")
    for modelname, model_config in model_configs['model_metadata'].items():
        model = models[modelname]['model_obj']
        test_df = models[modelname]['data']['test']
        results = model(test_df['imgpath'].tolist())
        results_dfs = [] 

        # pull the df result, also adding on the image path for each result

        for result in results:
            result_df = result.to_df()
            result_df['imgpath'] = result.path
            results_dfs.append(result_df)
        results_df = pd.concat(results_dfs, axis=0)
        results_df = pd.merge(results_df, test_df, on='imgpath', how='outer')
        # If there are any nan's, make a column for "detected" and set it to False
        results_df['is_detected'] = results_df['box'].apply(lambda x: False if pd.isna(x) else True)

        # Convert the results to the correct format, pulling out the box coordinates and converting them to the correct format. This pulls 'x1', 'y1', 'x2', 'y2' from the 'box' column. Also account for the fact that box may be a NaN value. Fill the nans with a dictionary of -1 values
        results_df['box'] = results_df['box'].apply(lambda x: {'x1': -1, 'y1': -1, 'x2': -1, 'y2': -1} if pd.isna(x) else x)
        results_df.rename(columns={'box': 'pred_box'}, inplace=True)
        results_df['pred_x1'] = results_df['pred_box'].apply(lambda x: x['x1'])
        results_df['pred_y1'] = results_df['pred_box'].apply(lambda x: x['y1'])
        results_df['pred_x2'] = results_df['pred_box'].apply(lambda x: x['x2'])
        results_df['pred_y2'] = results_df['pred_box'].apply(lambda x: x['y2'])
        # Pull out img width and height
        results_df['pred_obj_width'] = results_df['pred_y2'] - results_df['pred_y1']
        results_df['pred_obj_height'] = results_df['pred_x2'] - results_df['pred_x1']
        # Convert the box coordinates to the normalized format
        results_df['pred_ymin'] = results_df['pred_y1'] 
        results_df['pred_xmax'] = results_df['pred_x2'] 
        results_df['pred_xmin'] = results_df['pred_x1'] 
        results_df['pred_ymax'] = results_df['pred_y2'] 
        results_df['pred_xmin_norm'] = results_df['pred_xmin'] / results_df['img_width']
        results_df['pred_xmax_norm'] = results_df['pred_xmax'] / results_df['img_width']
        results_df['pred_ymin_norm'] = results_df['pred_ymin'] / results_df['img_height']
        results_df['pred_ymax_norm'] = results_df['pred_ymax'] / results_df['img_height']

        # Combine this with the test_df on the 'imgpath' column. We want to use an outer with results and test df, since there may be imagees with no detects or multiple detects
        models[modelname]['test_results'] = results_df
        makedirs(output_dir, exist_ok=True)
        results_df.to_csv(path.join(output_dir, f'{modelname}_test_results.csv'))
    return models

def validate_models(models, model_configs):
    """Validate all models

    Args:
        models (dict): Dictionary of model names to model objects
        model_configs (dict): Model configuration, loaded from yaml

    Returns:
        models: Dictionary of model names to model objects updated with validation results 
    """
    defaults = model_configs['default_configuration']
    for modelname, model_config in model_configs['model_metadata'].items():
        models[modelname]['validation_results'], models[modelname]['validation_df'] = validate_model(
            model=models[modelname]['model_obj'], 
            modelname=modelname,
            output_dir=models[modelname]['output_dir'],
            validation_args=model_config.get('validation_parameters', defaults['validation_parameters']),
            seed=model_config.get('seed', defaults['seed']),
            device=verify_device(model_config.get('device', defaults['device'])),
            interface_yaml=models[modelname]['interface_yaml'])
    
    return models
def validate_model(model, modelname, output_dir, interface_yaml, device, seed, validation_args):
    """Validate a single model
    Args:
        model (Model object): Model object to validate
        modelname (string): Name of the model
        output_dir (string, path-like): Output directory for the model 
        interface_yaml (string, path-like): Path to the datasets.yaml file
        device (string): Device to train on
        seed (int): Seed for reproducibility 
        validation_args (dict): Additional validation arguments 
    """
    if 'bypass_validation' in validation_args:
        if validation_args['bypass_validation']:
            logging.info("Bypassing validation")
            return None, None
        validation_args.pop('bypass_validation')
    logging.info(f"Validating model {modelname}")
    interface_yaml='/Users/thomasgilmore/Documents/20_Academic/ECE_Masters/ECE_5554_Computer_Vision/Project/license-plate-detection/src/../outputs/datasets/cars_license_plate_new/datasets.yaml'
    results = model.val(
        data=interface_yaml,
        project=f'outputs/{modelname}', # Set the output directory for the model
        name = f'{output_dir}/validation',  # Name of the experiment
        device=device,  # Device to train on
        seed=seed,
        **validation_args # Additional training arguments
    )
    results_df = pd.DataFrame()
    results_df['box_map'] = results.box.map # Mean AP at IoU thresholds of 0.5 to 0.95 for all classes
    results_df['box_map50'] = results.box.map50 # Mean AP at IoU threshold of 50%
    results_df['box_map75'] = results.box.map75 # Mean AP at IoU threshold of 75%
    results_df['mean_results'] = results.box.mean_results()
    results_df['fitness'] = results.box.fitness()
    return results, results_df


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
    device = verify_device(model_config.get('device', defaults['device']))
    # model_name = path.join(LPG.OUTPUTS_DIR, model_dir, + ".pt")

    # We need to look for the latest directory in the model directory matching the model name. This is denoted by the highest number at the end of the directory name. Assume that we are going to start at 0 
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
    # Split data 
    train_df, val_df, test_df, remaining_points = test_train_split(
        data, 
        train_split=train_split, 
        test_split=test_split, 
        validation_split=validation_split, 
        seed=model_config.get('seed', defaults['seed']),
        output_dir=model_dir,
        model=model
    )
    # Reformat to yaml format
    interface_yaml, new_train_df, new_val_df, new_test_df = createYamlFormattedData(train_df, val_df, test_df, model_dir, save=False)
    if model_config.get('load_saved_models', defaults['load_saved_model']):
        expected_dir = path.join(model_output_dir, str(latest_model_dir - 1))
        if path.exists(expected_dir) and path.exists(path.join(expected_dir, model + '.pt')) and path.exists(path.join(expected_dir, 'weights', 'best.onnx')):
            # Load the saved model from the model directory .pt file 
            saved_model_loc = path.join(expected_dir, model + '.pt')
            logging.info(f"Loading model {saved_model_loc}")
            saved_model = YOLO(saved_model_loc)

            best_model = YOLO(path.join(expected_dir, 'weights', 'best' + '.pt'))
            return {
                'model_obj' : saved_model, 
                'output_dir': expected_dir, 
                'train_results': None, 
                'best_model': best_model, 
                'interface_yaml': path.join(expected_dir, 'dataset', 'datasets.yaml'),
                'data': {
                    'train': new_train_df, 
                    'val': new_val_df, 
                    'test': new_test_df, 
                    'remaining': remaining_points, 
                    'original': data, 
                    'original_train': train_df, 
                    'original_val': val_df, 
                    'original_test': test_df
                }  
            }

            # Load the best model from the weights directory
            
            # Load the best weights from the weights directory
            # ret_model.load_weights(path.join(expected_dir, 'weights', 'best.onnx'))

    elif model_config.get('load_saved_models', defaults['load_saved_model']):
        logging.warning(f"You specified to load saved model, but model {model_name} does not exist. Training new model.")
        
    # At this point, we would want to save the data 
    interface_yaml, new_train_df, new_val_df, new_test_df = createYamlFormattedData(train_df, val_df, test_df, model_dir, save=True)
    plot_test_train_split(new_train_df, new_val_df, new_test_df, remaining_points, model_dir, model)
    logging.info(f"Training model {model_name}")
    model_obj = YOLO(model_config['model_type'])


    # Store the train, validation,model_co and test data in the model directory

    new_data_dir = path.join(LPG.OUTPUTS_DIR, 'datasets', 'cars_license_plate_new')

    

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
    return {
        'model_obj' : model_obj, 
        'train_results': train_results, 
        'output_dir': model_dir, 
        'best_model': model_obj, 
        'interface_yaml': interface_yaml,
        'data': {
            'train': new_train_df, 
            'val': new_val_df, 
            'test': new_test_df, 
            'remaining': remaining_points, 
            'original': data, 
            'original_train': train_df, 
            'original_val': val_df, 
            'original_test': test_df
        }  
    }

def test_train_split(data, train_split=0.8, test_split=0.1, validation_split=0.1,seed=None, output_dir=None, model=None):
    # Build test / train split 
    # Randomly pre-filter the data based on the train, validation, and test split. If the sum of the splits is less than 1, the remaining data will be unused.
    if train_split + test_split + validation_split > 1:
        raise ValueError("Sum of train, test, and validation splits must be less than or equal to 1")
    elif train_split + test_split + validation_split == 1:
        # Shuffle data 
        new_data = data.sample(frac=1, random_state=seed)
        # Make remaining data of size 0 with same columns as new_data
        remaining_points = pd.DataFrame(columns=new_data.columns)
    else: 
        new_data, remaining_points = train_test_split(data, test_size=1 - (train_split + test_split + validation_split), random_state=model_config.get('seed', defaults['seed']))
    # Since the split is based on the original size of the data, we need to scale the split sizes based on the new data size
    adj_train_split = train_split / (train_split + test_split + validation_split)
    adj_test_split = test_split / (test_split + validation_split)
    adj_validation_split = validation_split / (test_split + validation_split)
    train_df, tmp = train_test_split(new_data, test_size= 1 - adj_train_split, random_state=seed)
    logging.info("Sample of training data:")
    logging.info("\n" + train_df[['xmin', 'xmax', 'ymin', 'ymax', 'img_width', 'img_height']].head().to_string(index=False))
    val_df, test_df = train_test_split(tmp, test_size= 1 - adj_validation_split, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
  
    return train_df, val_df, test_df, remaining_points

def make_output_df(model_dict, out_dir): 
    # Create a dataframe with the model name, training results, and validation results
    data_dfs = []

    for modelname, model in model_dict.items():
        if 'train_df' in model and 'validation_df' in model:
            data_df = pd.concat([model['train_df'], model['validation_df']])
        elif 'train_df' in model:
            data_df = model['train_df']
        elif 'validation_df' in model:
            data_df = model['validation_df']
        else:
            data_df = pd.DataFrame()
        data_dfs.append(data_df)
    df = pd.concat(data_dfs)    
    df.to_csv(path.join(out_dir, 'model_output_metrics.csv'))
    return df

def plot_test_train_split(train_df, val_df, test_df, remaining_points, output_dir, model):
    # Create a pie chart using plotly for the sizes of the train, validation, and test df sizes  
    pie_chart(data=pd.DataFrame({'size': [len(train_df), len(val_df), len(test_df), len(remaining_points)], 'type': ['Train Split', 'Validation Split', 'Test Split', 'Remaining Data']}),
              values='size', names='type', title='Dataset Split', filename='dataset_split', output_dir=path.join(LPG.PLOTS_DIR, model))