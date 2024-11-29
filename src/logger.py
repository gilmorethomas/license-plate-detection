import logging
import colorlog
from os import path, makedirs
from src.globals import LICENSE_PLATE_GLOBALS as LPG

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

logger = colorlog.getLogger('example_logger')
logger.setLevel(logging.DEBUG)  # Set the logging level

# Check if handlers are already added to avoid duplicate handlers
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Prevent log messages from being propagated to the root logger
logger.propagate = False

def set_logfile(output_dir, name):
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(path.join(output_dir, name))
    
    # Set level and formatter for handlers
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
def setLevel(level, logger=logger):
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }    
    logger.setLevel(levels.get(level.upper(), logging.DEBUG))

set_logfile(path.join(path.abspath(LPG.OUTPUTS_DIR)), 'output.log')