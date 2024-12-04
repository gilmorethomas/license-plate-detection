import os 
from os import path
import string
import easyocr
from src.utilities import load_image, overlay_boxes, imshow, save_images, save_image
from src.plotting import histogram, bar_plot
import pytesseract
import cv2
import pandas as pd 
import numpy as np
from os import makedirs
# Initialize the OCR reader
from ultralytics import YOLO
from src.logger import logger as logging 
import Levenshtein as lev

reader = easyocr.Reader(['en'], gpu=False, detect_network='craft', recog_network='standard')

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
allowlist = string.ascii_uppercase + string.digits + string.ascii_lowercase + '.' + ' '
tesseract_config = {'psm': 6, 'oem' : 3, 'lang': 'eng', 'char_whitelist':f'{allowlist}'}
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



def read_license_plate(license_plate_crop, enforce_format=False, threshold=0.1, engine='tesseract'):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.
        enforce_format (bool): If True, enforce the license plate format.
        threshold (float): Confidence threshold for the OCR reader.

    Returns:
        list(tuple): List of tuples containing the license plate text and the corresponding score.
    """
    return_detections = []
    if engine == 'easyocr':
        detections = reader.readtext(license_plate_crop, allowlist=allowlist, rotation_info=[0])
        #, 90, 180, 270])

        for detection in detections:
            bbox, text, score = detection
            if score < threshold:
                continue
            if enforce_format: 
                text = text.upper().replace(' ', '')
                if license_complies_format(text):
                    return_detections.append([format_license(text), score])
            else:
                return_detections.append({'x1': bbox[0][0], 'y1': bbox[0][1], 'x2': bbox[2][0], 'y2': bbox[2][1], 'text': text, 'score': score})
        
    elif engine == 'tesseract': 

        # predicted_result = pytesseract.image_to_string(license_plate_crop, lang ='eng', config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') 
        predicted_result = pytesseract.image_to_string(license_plate_crop, lang =tesseract_config.get('lang', 'eng'), config =f"--oem {tesseract_config.get('oem', '3')} --psm {tesseract_config.get('psm', 7)} -c tessedit_char_whitelist={tesseract_config.get('char_whitelist', allowlist)}").strip()         
        data = pytesseract.image_to_data(license_plate_crop, lang =tesseract_config.get('lang', 'eng'), config =f"--oem {tesseract_config.get('oem', '3')} --psm {tesseract_config.get('psm', 7)} -c tessedit_char_whitelist={tesseract_config.get('char_whitelist', allowlist)}", output_type=pytesseract.Output.DICT)         
        # Change the 'conf' key to 'score' for consistency
        data['score'] = data.pop('conf')
        # Change left, top, width, height to x1, y1, x2, y2
        data['x1'] = data.pop('left')
        data['y1'] = data.pop('top')
        data['x2'] = [x + w for x, w in zip(data['x1'], data['width'])]
        data['y2'] = [y + h for y, h in zip(data['y1'], data['height'])]
        data.pop('width')
        data.pop('height')
        data.pop('level')
        data.pop('page_num')
        data.pop('block_num')
        data.pop('par_num')
        data.pop('word_num')
        data.pop('line_num')
        
        # Reformat dictionary of lists to be a list of dictionaries 
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        # Filter out any empty strings
        data = [d for d in data if d['text'] != '']
        # Filter out anything with a -1 score, since this means there is no confidence score available
        data = [d for d in data if d['score'] != -1]
        return_detections += data

    else: 
        raise ValueError('Invalid OCR engine. Please use either "easyocr" or "tesseract".')

    if len(return_detections) == 0:
        return_detections.append({'x1': None, 'y1': None, 'x2': None, 'y2': None, 'text': None, 'score': None})

    return pd.DataFrame(return_detections)


def overlay_recognition_results(img, results):
    """
    Overlay the recognition results on the image.
    
    """
    for result in results:
        # Make sure all of the values are not None
        if any([res is None for res in result.values()]):
            continue
        # Add anything that is not in x1, y1, x2, y2 to the additional_info key 
        additional_info = {key: value for key, value in result.items() if key not in ['x1', 'y1', 'x2', 'y2']}
        result = {key: value for key, value in result.items() if key in ['x1', 'y1', 'x2', 'y2']}
        result['color'] = (255, 0, 0)
        result['additional_info'] = additional_info
        img = overlay_boxes(img, result)
    return img

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

def preprocess_img(img):
    """ Preprocess the image for OCR. Implement the following steps:
        1. De-colorize the image (convert to grayscale)
        2. Posterize the image (convert to binary with threshold 64)
        3. Apply Gaussian blur to the image
        4. Convert the image to RGB format
        5. Invert the image (black text on white background)
    return: plate_gray, plate_treshold, img
    """
        # de-colorize
    plate_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # posterize
    _, plate_treshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

    # Applies Gaussian blur to the image
    img = cv2.GaussianBlur(plate_treshold, (5, 5), 0)
    # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format, we need to convert from BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Tesseract prefers non-inverted images (black text on white background)
    img = cv2.bitwise_not(img)
    return plate_gray, plate_treshold, img

def perform_ocr_of_single_img(img, bounding_box, out_dir, save_name='final_license_plate', save_image_debug=True):  
    """ Perform OCR on a single image. 
    """
    # Read the license plate from cropped truth image
    # Overlay the bounding boxes on the image to verify 
    img2 = overlay_boxes(img, bounding_box)
    # imshow([img, img2], "Combined Images", "matplotlib")

    # Crop the image to get the license plate
    license_plate_crop = img[int(bounding_box['y1']):int(bounding_box['y2']), int(bounding_box['x1']):int(bounding_box['x2'])]
    if license_plate_crop.size == 0:
        logging.warning(f"License plate crop for {save_name} is empty. Skipping OCR.")

        return pd.DataFrame({'x1': -1, 'y1': -1, 'x2': -1, 'text': 'N/A', 'score': np.nan}, index=[0])
    # Read the license plate from cropped truth image
    gray, plate_treshold, preprocessed_img = preprocess_img(license_plate_crop)

    # Save the plate_threshold image

    results = read_license_plate(preprocessed_img, enforce_format=False)
    if save_image_debug: 
        img_with_results_preprocessesd = overlay_recognition_results(preprocessed_img, results)
        save_images([img, license_plate_crop, img_with_results_preprocessesd], out_dir, save_name)
        save_image(plate_treshold, out_dir, save_name + '_treshold')
        save_image(gray, out_dir, save_name + '_gray')
        save_image(preprocessed_img, out_dir, save_name + '_preprocess')
    return results

def perform_ocr_on_df_images(model_dict, verbose=False): 
    """ Perform OCR on the images in the dataframe. 
    """
    logging.info("Performing OCR on the images in the dataframe.")
    for model_name, this_model_dict in model_dict.items():
        output_dir = path.join(this_model_dict['output_dir'], 'ocr_results')
        results = []

        for idx, row in this_model_dict['test_results'].iterrows():
            img = load_image(row['imgpath'])
            res = perform_ocr_of_single_img(img, row['pred_box'], output_dir, save_name=row['imgname'], save_image_debug=verbose)
            # rename res x1, y1, x2, y2 to ocr_text_x1, ocr_text_y1, ocr_text_x2, ocr_text_y2
            res = res.rename(columns={'x1': 'ocr_text_x1', 'y1': 'ocr_text_y1', 'x2': 'ocr_text_x2', 'y2': 'ocr_text_y2', 'text': 'ocr_text', 'score': 'ocr_score'})
            res['imgname'] = row['imgname']
            results.append(res) 
        these_results = pd.concat(results).reset_index(drop=True)
        these_results['imgname'] = these_results['imgname'].astype(str)
        this_model_dict['test_results']['imgname'] = this_model_dict['test_results']['imgname'].astype(str)
        
        # Merge this with the original dataframe
        merged_results = these_results.join(this_model_dict['test_results'], how='outer', rsuffix='_r')

        # Calculate the Levenshtein distance between the OCR text and the license plate text. If there is no license plate text (ocr_text or Plate Number is None or nan), set the distance to -1
        merged_results['levenshtein_distance'] = merged_results.apply(lambda x: lev.distance(str(x['ocr_text']), str(x['Plate_Number'])) if pd.notnull(x['ocr_text']) and pd.notnull(x['Plate_Number']) else -1, axis=1)
        # Calculate the Levenshtein similarity between the OCR text and the license plate text. If there is no license plate text (ocr_text or Plate Number is None or nan), set the similarity to -1
        merged_results['levenshtein_similarity'] = merged_results.apply(lambda x: lev.ratio(str(x['ocr_text']), str(x['Plate_Number'])) if pd.notnull(x['ocr_text']) and pd.notnull(x['Plate_Number']) else -1, axis=1)
        merged_results['ocr_correct'] = merged_results.apply(lambda x: x['ocr_text'] == x['Plate_Number'], axis=1)

        # Merge this with the original dataframeke
        makedirs(output_dir, exist_ok=True)
        merged_results.to_csv(path.join(output_dir, model_name + '_ocr_results.csv'), index=False)
        merged_results['model_name'] = model_name   
        results.append(merged_results)
        model_dict[model_name]['ocr_results'] = these_results
        create_ocr_performance_plots(merged_results, output_dir)
    return model_dict

def create_ocr_performance_plots(df, output_dir): 
    """ Create the OCR performance plots. 
    Args: 
        df (pd.DataFrame): The dataframe containing the OCR results.
    """
    logging.info("Creating OCR performance plots")
    # Create a histogram of the OCR scores
    histogram(df, x='ocr_score', title='OCR Score Distribution', filename='ocr_score_distribution', output_dir=output_dir, save_png=True, save_html=True)
    # Create a dataframe where the number of detections is counted. This is determined by the number of instances of the same image name
    detection_counts = df.groupby('imgname').size().reset_index(name='detection_count')
    # Create a histogram of the number of detections per image
    histogram(detection_counts, x='detection_count', title='Number of OCR Detections per Image', filename='ocr_detections_per_image', output_dir=output_dir, save_png=True, save_html=True, d_levels=[1])
    # Create a histogram of the top 10 most common OCR detections and least common OCR detections 
    # Get the top 10 most common detections
    top_10 = detection_counts.nlargest(10, 'detection_count')
    histogram(top_10, 'detection_count', 'Top 10 Most Common OCR Detections', 'ocr_top_10', output_dir, save_png=True, save_html=True)
    # Get the 10 least common detections
    bottom_10 = detection_counts.nsmallest(10, 'detection_count')
    histogram(bottom_10, 'detection_count', 'Top 10 Least Common OCR Detections', 'ocr_bottom_10', output_dir, save_png=True, save_html=True)
    # Calculate the accuracy of the OCR detections
    # Get the number of correct detections
    # Create a histogram of the number of correct detections. 
    histogram(df, 'ocr_correct', 'OCR Correct Distribution', 'ocr_correct_distribution', output_dir, save_png=True, save_html=True)

    # Calculate the accuracy of detections, given by the similarity of the OCR text to the license plate text. 
    # This is done using Levenshtein distance
    # Create a histogram of the Levenshtein distance
    histogram(df, 'levenshtein_distance', 'Levenshtein Distance Distribution', 'levenshtein_distance_distribution', output_dir, save_png=True, save_html=True, d_levels=[-1])
    # Create a histogram of the Levenshtein similarity
    histogram(df, 'levenshtein_similarity', 'Levenshtein Similarity Distribution', 'levenshtein_similarity_distribution', output_dir, save_png=True, save_html=True, d_levels=[-1])
    # Create a bar plot of the top 5 and bottom 5 Levenshtein distances on the same plot
    # Get the top 5 most common detections
    top_5 = df.nlargest(5, 'levenshtein_distance')
    bottom_5 = df.nsmallest(5, 'levenshtein_distance')
    top_bottom_5 = pd.concat([top_5, bottom_5])
    bar_plot(top_bottom_5, 'imgname', 'levenshtein_distance', 'Top 5 and Bottom 5 Levenshtein Distances', 'top_bottom_5_levenshtein_distance', output_dir, save_png=True, save_html=True)
    # Create a bar plot of the top 5 and bottom 5 Levenshtein similarities on the same plot
    top_5 = df.nlargest(5, 'levenshtein_similarity')
    bottom_5 = df.nsmallest(5, 'levenshtein_similarity')
    top_bottom_5 = pd.concat([top_5, bottom_5])
    bar_plot(top_bottom_5, 'imgname', 'levenshtein_similarity', 'Top 5 and Bottom 5 Levenshtein Similarities', 'top_bottom_5_levenshtein_similarity', output_dir, save_png=True, save_html=True)
    # Create a bar plot of the top 5 and bottom 5 ocr scores 
    df_ocr_scores = df.dropna(subset=['ocr_score'])
    df_ocr_scores = df_ocr_scores[df_ocr_scores['ocr_score'] != 'nan']
    df_ocr_scores['ocr_score'] = df_ocr_scores['ocr_score'].astype(float)
    top_5 = df_ocr_scores.nlargest(5, 'ocr_score')
    bottom_5 = df_ocr_scores.nsmallest(5, 'ocr_score')
    top_bottom_5 = pd.concat([top_5, bottom_5])
    bar_plot(top_bottom_5, 'imgname', 'ocr_score', 'Top 5 and Bottom 5 OCR Scores', 'top_bottom_5_ocr_scores', output_dir, save_png=True, save_html=True)


    return df    

if __name__ == "__main__":
    # # Load the image and labels
    # # Get path of this file 
    dirname = os.path.dirname(__file__)
    # img = load_image(path.join(dirname ,'../test_data/Cars6.png'))
    # labels = pd.read_csv(path.join(dirname, '../test_data/Cars6.txt'), delimiter=' ', header=None)
    # simple_detector  = YOLO(path.join(dirname, '..', 'test_data', 'simple_model.pt'))
 
    # # Rename the columns
    # labels.columns = ['class_id', 'x_center', 'y_center', 'width', 'height']
    # # Get the image dimensions
    # img_height, img_width, _ = img.shape
    # labels['img_width'] = img_width
    # labels['img_height'] = img_height
    # # Calculate the bounding box coordinates in non-normalized format. This is the inverse of what is done in the YOLO format conversion
    # labels['xmin'] = (labels['x_center'] - labels['width'] / 2) * labels['img_width']
    # labels['xmax'] = (labels['x_center'] + labels['width'] / 2) * labels['img_width']
    # labels['ymin'] = (labels['y_center'] - labels['height'] / 2) * labels['img_height']
    # labels['ymax'] = (labels['y_center'] + labels['height'] / 2) * labels['img_height']



    # x_center = (labels['xmin'] + labels['xmax']) / 2 / labels['img_width']
    # y_center = (labels['ymin'] + labels['ymax']) / 2 / labels['img_height']
    # width = (labels['xmax'] - labels['xmin']) / labels['img_width']
    # height = (labels['ymax'] - labels['ymin']) / labels['img_height'] 
    # box_dict = {}
    # box_dict['x1'] = labels['xmin'].iloc[0]
    # box_dict['x2'] = labels['xmax'].iloc[0]
    # box_dict['y1'] = labels['ymin'].iloc[0]
    # box_dict['y2'] = labels['ymax'].iloc[0]
    # results = perform_ocr_of_single_img(img, box_dict, path.join(dirname, '..', 'test_data'), save_name='license_plate_6')
    # # Output to CSV with pandas dataframe 
    # df = pd.DataFrame(results)
    # df.to_csv(path.join(dirname, '..', 'test_data', 'license_plate_6.csv'), index=False)
    # # Do the same as the above for license plate 210, which has reversed text 
    # # Load the image and labels
    # # Get path of this file 
    # dirname = os.path.dirname(__file__)
    # img = load_image(path.join(dirname ,'../test_data/Cars210.png'))
    # labels = pd.read_csv(path.join(dirname, '../test_data/Cars210.txt'), delimiter=' ', header=None)
    # simple_detector  = YOLO(path.join(dirname, '..', 'test_data', 'simple_model.pt'))
 
    # # Rename the columns
    # labels.columns = ['class_id', 'x_center', 'y_center', 'width', 'height']
    # # Get the image dimensions
    # img_height, img_width, _ = img.shape
    # labels['img_width'] = img_width
    # labels['img_height'] = img_height
    # # Calculate the bounding box coordinates in non-normalized format. This is the inverse of what is done in the YOLO format conversion
    # labels['xmin'] = (labels['x_center'] - labels['width'] / 2) * labels['img_width']
    # labels['xmax'] = (labels['x_center'] + labels['width'] / 2) * labels['img_width']
    # labels['ymin'] = (labels['y_center'] - labels['height'] / 2) * labels['img_height']
    # labels['ymax'] = (labels['y_center'] + labels['height'] / 2) * labels['img_height']



    # x_center = (labels['xmin'] + labels['xmax']) / 2 / labels['img_width']
    # y_center = (labels['ymin'] + labels['ymax']) / 2 / labels['img_height']
    # width = (labels['xmax'] - labels['xmin']) / labels['img_width']
    # height = (labels['ymax'] - labels['ymin']) / labels['img_height'] 
    # box_dict = {}
    # box_dict['x1'] = labels['xmin'].iloc[0]
    # box_dict['x2'] = labels['xmax'].iloc[0]
    # box_dict['y1'] = labels['ymin'].iloc[0]
    # box_dict['y2'] = labels['ymax'].iloc[0]
    # results = perform_ocr_of_single_img(img, box_dict, path.join(dirname, '..', 'test_data'), save_name='license_plate_210')
    # df = pd.DataFrame(results)
    # df.to_csv(path.join(dirname, '..', 'test_data', 'license_plate_210.csv'), index=False)

    # # Perform OCR on the entire dataframe
    ocr_results = pd.read_csv(path.join(dirname, '..', 'test_data', 'model_a_ocr_results.csv'))
    create_ocr_performance_plots(ocr_results, path.join(dirname, '..', 'test_data', 'ocr_results'))
    # ocr_results = perform_ocr_on_df_images(ocr_results, path.join(dirname, '..', 'test_data', 'ocr_results'), verbose=True)