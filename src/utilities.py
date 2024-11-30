from torchvision import datasets, transforms, utils
import os
from os import path 
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
import plotly.express as px
import numpy as np 
import cv2
from copy import deepcopy
from os.path import basename
import random
import torch
import numpy as np
from torch.backends import mps
from torch import cuda 
from src.logger import logger as logging 
import threading
import time
import psutil
from src.plotting import finalize_plot
from src.globals import LICENSE_PLATE_GLOBALS as LPG


def imshow(img, title=None, render_type="cv2"):
    """Displays images

    Args:
        img (list of imges or single image): Image(s) to display. If multiple images are passed, they are displayed in a grid.
        title (str, optional): Figure title. Defaults to None.
        render_type (str, optional): Whether to use matplotlib or cv2 for image display. Defaults to "matplotlib".

    Raises:
        ValueError: If an invalid render type is passed
    """
    # If multiple images are passed, display them in a grid
    try:     
        if isinstance(img, list):
            if render_type == "matplotlib":
                # Use matplotlib to show images in a grid
                num_images = len(img)
                cols = int(np.ceil(np.sqrt(num_images)))
                rows = int(np.ceil(num_images / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
                axes = np.array(axes)  # Ensure axes is a numpy array
                axes = axes.reshape(rows, cols)  # Reshape to 2D array if necessary
                for i, image in enumerate(img):
                    ax = axes[i // cols, i % cols] if rows > 1 and cols > 1 else axes[i % cols]
                    cv2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    ax.axis('off')
                if title:
                    fig.suptitle(title)
                plt.show()
            elif render_type == "cv2":
                # Combine multiple images into a single image. Need to convert to tensors first
                img = [transforms.ToTensor()(i) for i in img]
                # Account for the fact that images may not be the same size and pad with zeros
                max_height = max([i.shape[1] for i in img])
                max_width = max([i.shape[2] for i in img])
                # Pad with zeros to make all images the same size in height and width
                img = [transforms.Pad((0, 0, max_width - i.shape[2], max_height - i.shape[1]))(i) for i in img]

                img = utils.make_grid(img)
                cv2.imshow(title, cv2.cvtColor(np.transpose(img.numpy(), (1, 2, 0)), cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Invalid render type. Use either 'matplotlib' or 'cv2'")
        else:
            if render_type == "matplotlib":
                # Use matplotlib to show a single image
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                if title:
                    plt.title(title)
                plt.show()
            elif render_type == "cv2":
                cv2.imshow(title, img)
                cv2.waitKey(0)
            else:
                raise ValueError("Invalid render type. Use either 'matplotlib' or 'cv2'")
    except Exception as e:
        logging.error(f"Error displaying image: {e}")

def imshow_from_path(img_path, render_type="cv2"):
    # Use cv2 to show image 
    img = cv2.imread(img_path)
    imshow(img, title=basename(img_path).split('.')[0], render_type=render_type)

def load_image(img_path):
    assert path.exists(img_path), f"Image path {img_path} does not exist"
    return cv2.imread(img_path)

def overlay_boxes(img, box_list_dict):
    # Copy the image so we don't modify the original
    img_new = deepcopy(img)
    if not isinstance(box_list_dict, list):
        box_list_dict = [box_list_dict]
    for box in box_list_dict:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        # Cast to int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = box.get("color", (0, 255, 0))
        line_thickness = box.get("line_thickness", 2)
        # line_type = box.get("line_type", cv2.LINE_AA)

        # img = cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=line_thickness, lineType=line_type)
        img_new = cv2.rectangle(img=img_new, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=line_thickness)
        # Add additional text from the additional_info key
        additional_info = box.get("additional_info", {})
        if additional_info:
            text = ', '.join([f"{k}: {v}" for k, v in additional_info.items()])
            img_new = cv2.putText(img_new, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_new

def save_image(img, outdir, name): 
    """Writes an image to disk

    Args:
        img (np.Array): Image to write
        outdir (str, path-like): Output directory
        name (str): Name of the image
    """
    if not path.exists(outdir):
        os.makedirs(outdir)
    # If we don't have the extension at the end of the file, add it
    if not name.endswith('.png') or not name.endswith('.jpg'):
        name = name + '.png'
    cv2.imwrite(path.join(outdir, name), img)
    
def save_images(imgs, outdir, name):
    """Writes a list of images to disk. Combines the images into a single image

    Args:
        imgs (list[np.Array]): List of images to write
        outdir (str, path-like): Output directory
        names (list[str]): List of names of the images
    """
    if not path.exists(outdir):
        os.makedirs(outdir)
    if not isinstance(imgs, list):
        imgs = [imgs]
    # Combine the images into a single image
    img = [transforms.ToTensor()(i) for i in imgs]
    max_height = max([i.shape[1] for i in img])
    max_width = max([i.shape[2] for i in img])
    # Pad with zeros to make all images the same size in height and width
    img = [transforms.Pad((0, 0, max_width - i.shape[2], max_height - i.shape[1]))(i) for i in img]
    # Use torch stack to combine the images

    img_new = utils.make_grid(img)
    # Convert to 0-255 range
    img_new = img_new.mul(255).byte()
    # Need to make sure the image is in the right format
    img_new = np.transpose(img_new.numpy(), (1, 2, 0))
    save_image(img_new, outdir, name)

def show_multiple_images(imgs):
    imshow(utils.make_grid(imgs))

if __name__ == '__main__':
    assumed_datasetpath = path.join(os.path.expanduser("~"), ".cache/kagglehub/datasets/andrewmvd/car-plate-detection/versions/1")

    img = load_image(path.join(assumed_datasetpath, "images/Cars0.png"))
    # imshow_from_path(path.join(assumed_datasetpath, "images/Cars0.png"))
    coords = [
        {'x1': 226, 'y1': 125, 'x2': 419, 'y2': 173, "color": (0, 255, 0)},
        {'x1': 250, 'y1': 142, 'x2': 400, 'y2': 150, "color": (0, 0, 255)},
        ]

    img_new = overlay_boxes(img, coords)
    imshow([img, img_new], "Combined Images")

    imshow_from_path(path.join(assumed_datasetpath, "images/Cars1.png"))
    imshow_from_path(path.join(assumed_datasetpath, "images/Cars2.png"))
    imshow_from_path(path.join(assumed_datasetpath, "images/Cars3.png"))

    cv2.waitKey(0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def verify_device(device):
    """_summary_

    Args:
        device (_type_): _description_

    Returns:
        _type_: _description_
    """

    if device == 'mps':
        # Check that MPS is available
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled. Defaulting to CPU")
                device = 'cpu'
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
                logging.warning("MPS device not available. Falling back to CPU.")
                device = 'cpu'
    elif device == 'mps': 
        # Allow for MPS fallback, since not all operations are enabled in MPS
        logging.warning("MPS device selected. Enabling MPS fallback. Note that you need to set this before a torch import")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    elif device == 'cuda' and not cuda.is_available():
        logging.warning("CUDA device not available. Falling back to CPU.")
        device = 'cpu'
    elif device == 0 or device == 1: 
        logging.warning('Selected GPU. If not available, this will be converted on the backend')
    else:
        device = device
    return device 

def start_resource_utilization_thread(): 
    """Starts a thread to log resource utilization

    Returns:
    tuple
        A tuple containing: 
            stop_event : threading.Event 
                Event to stop the logging thread
            logging_thread: threading.Event
                Thread to log resource utilization
            cpu_usage: list[double]
                List of CPU usage
            memory_usage: list[double]
                List of memory usage
            gpu_usage: list[double]
                List of GPU usage
            timestamps: list[double]
                List of timestamps
    """
    cpu_usage = []
    gpu_usage = []
    memory_usage = []
    timestamps = []
    stop_event = threading.Event()
    logging_thread = threading.Thread(target=log_resource_utilization, args=(cpu_usage, memory_usage, gpu_usage, timestamps, stop_event,))
    logging_thread.start()
    return stop_event, logging_thread, cpu_usage, memory_usage, gpu_usage, timestamps

def log_resource_utilization(cpu_usage, memory_usage, gpu_usage, timestamps, stop_event):
    """Callback function to log resource utilization


    Args:
        cpu_usage (list[double]): List of CPU usage
        memory_usage (list[double]): List of memory usage
        gpu_usage (list[double]): List of GPU usage
        timestamps (list[double]): List of timestamps, relative to start of logging
        stop_event (threading.Event): Event to stop the logging thread
    """
    while not stop_event.is_set():
        cpu_usage.append(psutil.cpu_percent())
        memory_usage.append(psutil.virtual_memory().percent)
        gpu_usage.append(torch.cuda.memory_allocated())
        timestamps.append(time.time())
        time.sleep(1)
    logging.info("Resource utilization thread stopped")

def stop_resource_utilization_thread(stop_event, logging_thread): 
    stop_event.set()
    logging_thread.join()
    stop_event = threading.Event()

def plot_resource_utilization(cpu_usage, memory_usage, gpu_usage, timestamps, output_dir):
    """Plots resource utilization throughout run 
    Args:
        cpu_usage (list[double]): List of CPU usage
        memory_usage (list[double]): List of memory usage
        gpu_usage (list[double]): List of GPU usage
        timestamps (list[double]): List of timestamps, relative to start of logging
        output_dir (str, path-like): Output directory to save plots
    """
    # Convert timestamps to relative time, relative to min timestamp
    timestamps = [t - timestamps[0] for t in timestamps]
    fig = pgo.Figure()
    fig.add_trace(pgo.Scatter(x=timestamps, y=cpu_usage, mode='lines', name='CPU Usage'))
    fig.add_trace(pgo.Scatter(x=timestamps, y=memory_usage, mode='lines', name='Memory Usage'))
    fig.add_trace(pgo.Scatter(x=timestamps, y=gpu_usage, mode='lines', name='GPU Usage'))
    fig.update_layout(title='Resource Utilization', xaxis_title='Time (s)', yaxis_title='Percentage')
    # Normalize y axis scale to 100 
    fig.update_yaxes(range=[0, 100])
    logging.info(f"Creating resource utilization plot to {output_dir}")
    finalize_plot(fig, "Resource Utilization", "resource_utilization", path.join(output_dir, 'resource_utilization'), save_png=LPG.SAVE_PNG, save_html=LPG.SAVE_HTML)


def convert_xy_bounds_to_centered_xywh(df):
    """Converts the x, y bounds to the center x, y and width, height

    Args:
        df (pd.DataFrame): DataFrame containing the x, y bounds

    Returns:
        pd.DataFrame: DataFrame containing the center x, y and width, height
    """
    df['x_center'] = (df['xmin'] + df['xmax']) / 2 / df['img_width']
    df['y_center'] = (df['ymin'] + df['ymax']) / 2 / df['img_height']
    df['width'] = (df['xmax'] - df['xmin']) / df['img_width']
    df['height'] = (df['ymax'] - df['ymin']) / df['img_height']
    return df