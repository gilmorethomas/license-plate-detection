import comet_ml

from src.logger import logger as logging 
from src.pulldataset import download_dataset, parse_dataset, PullInPlateTruth
from src.utilities import verify_device, start_resource_utilization_thread, stop_resource_utilization_thread, plot_resource_utilization
from src.model import train_models, validate_models, test_models, make_output_df
import yaml
import shutil
import os
from os import path, makedirs
from src.plotting import pie_chart
import pandas as pd 
from src.globals import LICENSE_PLATE_GLOBALS as LPG 
import time
from inspect import signature, Parameter


def configure_comet_ml():
    """Configure Comet ML
    """
    # Configure Comet ML
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
    data = PullInPlateTruth(data)

    # Create test and train splits

    with open("inputs/model_config.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    model_dict = train_models(model_config, data)
    model_dict = validate_models(
        models=model_dict, 
        model_configs=model_config,
    )
    # Log the model dict to csv 
    model_dict = test_models(
        models=model_dict, 
        model_configs=model_config,
    )
    make_output_df(model_dict, LPG.OUTPUTS_DIR)
    # model_df = pd.DataFrame(model_dict)
    # model_df.to_csv(path.join(LPG.OUTPUTS_DIR, 'models.csv'))

    stop_resource_utilization_thread(stop_event, logging_thread)
    plot_resource_utilization(cpu_usage, memory_usage, gpu_usage, timestamps, LPG.PLOTS_DIR)