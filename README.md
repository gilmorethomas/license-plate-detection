# license-plate-detection
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
* Run main.py with the sample video file to generate the test.csv file 
``` python
python main.py
```
* Run the add_missing_data.py file for interpolation of values to match up for the missing frames and smooth output.
```python
python add_missing_data.py

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
* https://mpolinowski.github.io/docs/IoT-and-Machine-Learning/ML/2023-09-15--yolo8-tracking-and-ocr/2023-09-15/
* https://docs.ultralytics.com/modes/train/
* YOLOv8 architecture overview https://www.mdpi.com/2227-7080/12/9/164 

## Main components of this include 
1. Datset preparation 
    - Collect and process dataset of images to detect license plates on cars (YOLOv8)
2. Object detection
    - Object detector implemented using YOLOv8
2. Classification using Oriented Bounding Boxes (OBB) 
3. Logging
   * Logging is done using a combination of custom code as well as Comet. Comet is an Ultralytics product that allows for server-based logging 
   * You can turn of logging with Comet by setting USE_COMET in globals.py to be false
   * TODO better capture this 

## Potential future updates 
* Integration with Autodistill 
    * https://github.com/autodistill/autodistill?ref=blog.roboflow.com
    * https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/ 
* Integration with comet for logging 
* Integration with SORT, which offers realtime tracking algorithm for multiple object tracking in videos, using rudimentary association and state estimation 