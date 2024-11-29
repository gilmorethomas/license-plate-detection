import os 
from os import path
import string
import easyocr
from src.utilities import load_image, overlay_boxes, imshow
import pandas as pd 
# Initialize the OCR reader
from ultralytics import YOLO

reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

if __name__ == "__main__":
    # Load the image and labels
    # Get path of this file 
    dirname = os.path.dirname(__file__)
    img = load_image(path.join(dirname ,'../test_data/Cars6.png'))
    labels = pd.read_csv(path.join(dirname, '../test_data/Cars6.txt'), delimiter=' ', header=None)
    simple_detector  = YOLO(path.join(dirname, '..', 'test_data', 'simple_model.pt'))
 
    # Rename the columns
    labels.columns = ['class_id', 'x_center', 'y_center', 'width', 'height']
    # Get the image dimensions
    img_height, img_width, _ = img.shape
    labels['img_width'] = img_width
    labels['img_height'] = img_height
    # Calculate the bounding box coordinates in non-normalized format. This is the inverse of what is done in the YOLO format conversion
    labels['xmin'] = (labels['x_center'] - labels['width'] / 2) * labels['img_width']
    labels['xmax'] = (labels['x_center'] + labels['width'] / 2) * labels['img_width']
    labels['ymin'] = (labels['y_center'] - labels['height'] / 2) * labels['img_height']
    labels['ymax'] = (labels['y_center'] + labels['height'] / 2) * labels['img_height']



    x_center = (labels['xmin'] + labels['xmax']) / 2 / labels['img_width']
    y_center = (labels['ymin'] + labels['ymax']) / 2 / labels['img_height']
    width = (labels['xmax'] - labels['xmin']) / labels['img_width']
    height = (labels['ymax'] - labels['ymin']) / labels['img_height'] 
    box_dict = {}
    box_dict['x1'] = labels['xmin'].iloc[0]
    box_dict['x2'] = labels['xmax'].iloc[0]
    box_dict['y1'] = labels['ymin'].iloc[0]
    box_dict['y2'] = labels['ymax'].iloc[0]

    # Overlay the bounding boxes on the image to verify 
    img2 = overlay_boxes(img, box_dict)
    imshow([img, img2], "Combined Images", "matplotlib")

    # Crop the image to get the license plate
    license_plate_crop = img[int(box_dict['y1']):int(box_dict['y2']), int(box_dict['x1']):int(box_dict['x2'])]
    imshow([license_plate_crop], "License Plate Crop")
    imshow([img, license_plate_crop], "Combined Images")
    model_obj = YOLO('yolov8m')
    
    # Assume that we know all the cars and already have those images
    license_plate_detects = simple_detector(license_plate_crop)

    img_overlayed = img 
    for license_plate_detect in license_plate_detects: 
        for license_plate in license_plate_detect.boxes.data.tolist():
            # Overlay the bounding boxes on the image to verify
            license_plate_dict = {'x1': license_plate[0], 'y1': license_plate[1], 'x2': license_plate[2], 'y2': license_plate[3]}
            img_overlayed = overlay_boxes(img_overlayed, license_plate_dict)
            x1, y1, x2, y2, score, class_id = license_plate

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            imshow([license_plate_crop], "License Plate Crop")
            license_number, license_number_score = read_license_plate(license_plate_crop)
            print(license_number, license_number_score)
            if license_number is not None:
                print("License Plate Number: ", license_number)
            else:
                print("License Plate Number: ", "Not Found")
    
    imshow([img_overlayed], "License Plate Detection Model", "matplotlib")

    import pdb; pdb.set_trace()


    # Get the license plate coordinates    
    # Test the OCR reader
    # license_plate_crop = Image.open('data/license_plate.png')
