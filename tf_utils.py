import os
import glob as glob
import numpy as np

import sys
sys.path.append('/Users/rishabhbhardwaj/Downloads/tf_utils/models/research/object_detection')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

ALL_MODELS = ['faster_rcnn_inception_resnet_v2_atrous', 'faster_rcnn_resnet_101', 'faster_rcnn_inception_v2',
             'rfcn_resnet101', 'ssd_inception_v2', 'ssd_mobilenet_v1', 'yolo_v2']
MODEL_NAME = ALL_MODELS[0]
MODEL_PATH = os.path.join('object_detection/models/', MODEL_NAME)
PATH_TO_CKPT = os.path.join(MODEL_PATH,'inference_graph/frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('gtsdb_data', 'gtsdb3_label_map.pbtxt')
NUM_CLASSES = 3

PATH_TO_TEST_IMAGES_DIR = 'data/test'
PATH_TO_OUTPUT_DIR = 'gtsdb_data/'+MODEL_NAME+'/outputs'
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.png'))

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#print(label_map)

if not os.path.exists(PATH_TO_OUTPUT_DIR):
    os.makedirs(PATH_TO_OUTPUT_DIR)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
