from os import path 
class LICENSE_PLATE_GLOBALS:
    OUTPUTS_DIR = path.join(path.abspath(path.dirname(__file__)), '..', 'outputs')
    MODEL_DIR = path.join(OUTPUTS_DIR, 'models')
    DATA_DIR = path.join(OUTPUTS_DIR, 'data')
    LOG_DIR = path.join(OUTPUTS_DIR, 'logs')
    PLOTS_DIR = path.join(OUTPUTS_DIR, 'plots')
    SAVE_HTML = True
    SAVE_PNG = True
