from os import path, getenv
class LICENSE_PLATE_GLOBALS:
    OUTPUTS_DIR = path.join(path.abspath(path.dirname(__file__)), '..', 'outputs')
    MODEL_DIR = path.join(OUTPUTS_DIR, 'models')
    DATA_DIR = path.join(OUTPUTS_DIR, 'data')
    LOG_DIR = path.join(OUTPUTS_DIR, 'logs')
    PLOTS_DIR = path.join(OUTPUTS_DIR, 'plots')
    SAVE_HTML = True
    SAVE_PNG = True
    USE_COMET = True  # Set to True to use Comet.ml. This adds additional logging capability and visualization tools. Note you must have a comet account and API key to use this feature
    COMET_CONFIG = path.join(getenv('HOME'))
    PROJECT_NAME = "license-plate-detection"