# NVDA_DETEC
# Infrastructure for Real-Time Edge Object Reidentification and Tracking




Implementation of YOLO v3 object detector in Tensorflow (TF-Slim). Full tutorial can be found [here](https://medium.com/@pawekapica_31302/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe).

Tested on Python 3.5, Tensorflow 1.11.0 on Ubuntu 16.04.

## How to run the demo:
To run demo type this in the command line:

1. Download COCO class names file: `wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names`
2. Download and convert model weights:    
    1. Download binary file with weights from https://pjreddie.com/darknet/yolo/
    2. Run `python ./convert_weights.py`
3. Run `python ./demo.py --input_img <path-to-image> --output_img <name-of-output-image>`

YOLOv3 weights will need to be downloaded separately




## Changelog:
#### 2019-02-04: 

-Moved files to GitHub 

-YOLOv3 weights removed due to size


