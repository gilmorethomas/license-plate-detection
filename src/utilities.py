from torchvision import datasets, transforms, utils
import os
from os import path 
import matplotlib.pyplot as plt
import numpy as np 
import cv2
from copy import deepcopy
from os.path import basename
# The following is a substitute for cv2.imshow,
#  which you would use on your local machine but Colab does not support it

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
    for box in box_list_dict:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        color = box.get("color", (0, 255, 0))
        line_thickness = box.get("line_thickness", 2)
        # line_type = box.get("line_type", cv2.LINE_AA)

        # img = cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=line_thickness, lineType=line_type)
        img_new = cv2.rectangle(img=img_new, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=line_thickness)
    return img_new

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
    x = 0 
