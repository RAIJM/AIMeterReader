import jsonpickle
import numpy as np
import cv2
import os
import tensorflow as tf
import zipfile
import sys

from collections import defaultdict
from matplotlib import pyplot as plt
from PIL import Image

import matplotlib.pyplot as plt


# sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

MODEL_NAME = 'digital_meter_model'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'digitalmeter_label_map.pbtxt')

NUM_CLASSES = 10

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, file) 
                            for file in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)






def detect_objects(image_np,sess,detection_graph):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

   # rect_points, class_names, class_colors = draw_boxes_and_labels(
   #      boxes=np.squeeze(boxes),
   #      classes=np.squeeze(classes).astype(np.int32),
   #      scores=np.squeeze(scores),
   #      category_index=self.category_index,
   #      min_score_thresh=.5
   #  )

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    #plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()
    return image_np


def main():
    count =0;
    #for image_path in TEST_IMAGE_PATHS:
    image = Image.open('test_images/image11.jpg')
    image = image.rotate(90)
    image_np = load_image_into_numpy_array(image)
    image_np = detect_objects(image_np,sess,detection_graph)
    #im = Image.fromarray(image_np)
    #im.save(str(count)+".jpg")
    #count+=1



if __name__ == '__main__':
    main()
