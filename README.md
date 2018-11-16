# Red Round Sign Detection (and Classification)

## Setup

    1. Setup the environment : Run` pip install -r requirements.txt` to set up env

## Download Pre-Process Data

    1. Run `sh setup.sh` to download the data
    2. Download the gtsdb tfrecords created from [here](https://drive.google.com/file/d/1hKdjTsiFm_vdtZPdto0QJihThGFOPEkq/view)
    3. Run convert_format.py to create the test files for the notebook

## Training

The code uses Tensorflow Object Detection API (which is included in the repo), the instructions to run code can be found [here](https://github.com/rishabhbhardwaj/SignDetection/blob/master/Part2-Training.ipynb)

## References

  1. https://github.com/tensorflow/models/tree/master/research/object_detection
  2. https://github.com/aarcosg/traffic-sign-detection
