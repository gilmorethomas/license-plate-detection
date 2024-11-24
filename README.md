# license-plate-detection
License Plate Detection with YOLOv8 

## Project Setup

* Make an environment with python=3.10 using the following command 
``` bash
conda create --prefix ./liplad-with-yolov8 python==3.11 -y
```
* Activate the environment
``` bash
source activate ./liplad-with-yolov8 
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
* https://keylabs.ai/blog/getting-started-with-yolov8-a-beginners-guide/