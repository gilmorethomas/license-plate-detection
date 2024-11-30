\# license-plate-detection
License Plate Detection with YOLOv8 

## Project Setup

* Make an environment with python=3.10 using the following command 
``` bash
conda create --prefix ./liplad-with-yolov8 python==3.11.10 -y
```
* Activate the environment
``` bash
source activate ./liplad-with-yolov8 
``` 
* For macs, not all MPS operations are enabled by PyTorch. If you want to use the MPS in devices for training, you must enable MPS fallback to use the CPU. This can be done with the following command. If you have issues with this, you may see an exception raised during execution with Pytorch functions
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```
* You will need to install tesseract: 
Follow the instructions here https://tesseract-ocr.github.io/tessdoc/Installation.html

* Run main.py with the sample video file to generate the test.csv file 
``` python
python main.py
```


## Repo Structure
* Documentation: Includes report webpage 
* Resources: Additional resources that are referenced 
* src: Code 

Repo structure 
## References 
* https://docs.ultralytics.com/usage/python/#how-do-i-train-a-custom-yolo11-model-using-my-dataset
* https://www.kaggle.com/code/myriamgam62/car-plate-detection-yolov8-s This was used for pulling the dataset and xml schema
* https://keylabs.ai/blog/getting-started-with-yolov8-a-beginners-guide/
* Example for license plate detection and tracking: https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-09-15--yolo8-tracking-and-ocr/2023-09-15/
* https://docs.ultralytics.com/modes/train/
* YOLOv8 architecture overview https://www.mdpi.com/2227-7080/12/9/164 

## Main components of this include 
1. Datset preparation 
    - Collect and process dataset of images to detect license plates on cars (YOLOv8)
2. Object detection
    - Object detector implemented using YOLOv8
    - Model training: https://docs.ultralytics.com/modes/train/
    - Model validation: https://docs.ultralytics.com/modes/val/ 
3. Optical Character Recognition (OCR) using EasyOCR or Tesseract 
    - Images are pre-processed before fed to OCR to convert to greyscale. Thresholding is applied as well 
    - Bounding boxes of license plates are fed to OCR model
    - Limitations
        - License plates of 
    - https://www.jaided.ai/easyocr/
    - https://medium.com/geekculture/tesseract-ocr-understanding-the-contents-of-documents-beyond-their-text-a98704b7c655
    - https://pypi.org/project/easyocr/ 
    - https://www.jaided.ai/easyocr/documentation/ 
    - https://github.com/ankandrew/fast-plate-ocr
    - Using pre-trained English model 
    - There is currently no template-matching for plates
    - Detection model: Character Region Awareness for Text Detection (CRAFT)
        - https://arxiv.org/pdf/1904.01941
        - Defaults:
            - Text confidence threshold of 0.7
            - Text low-bound score of 0.4
            - Link Confidence threshold of 0.4
            - Maximum canvas size of 2560
            - Image magnification ratio of 1
    - Recognition network: https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md
        - No custom recognition network is used 
    - Compassing of images (rotations of 0, 90, 180, and 270 used)
3. Logging
   * Logging is done using a combination of custom code as well as Comet. Comet is an Ultralytics product that allows for server-based logging 
   * You can turn of logging with Comet by setting USE_COMET in globals.py to be false
   * TODO better capture this 

## Potential future updates 
* Integration with Autodistill 
    * https://github.com/autodistill/autodistill?ref=blog.roboflow.com
    * https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/ 
* Integration with comet for logging 
* Integration with SORT, which offers realtime tracking algorithm for multiple object tracking in videos, using rudimentary association and state estimation. This would allow for multiple looks at license plates, which would potentially help build greater confidence in the detected license plates and numbers
* Integration with TensorFlow for better logging