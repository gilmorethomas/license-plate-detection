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
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

def set_logfile(output_dir, name):
            # Create handlers
            c_handler = logging.StreamHandler()
            if not path.exists(output_dir):
                makedirs(output_dir)
            f_handler = logging.FileHandler(path.join(output_dir, name))
            c_handler.setLevel(logging.INFO)
            f_handler.setLevel(logging.INFO)

            # Create formatters and add it to handlers
            c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(c_format)
            f_handler.setFormatter(f_format)

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