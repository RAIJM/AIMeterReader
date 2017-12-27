from flask import Flask, request, make_response,Response,render_template,json,redirect, url_for
import os
import io
import numpy as np
import cv2
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
import imutils

from utility import *

sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

MODEL_NAME = 'digital_meter_model'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'digitalmeter_label_map.pbtxt')

NUM_CLASSES = 90


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

font = cv2.FONT_HERSHEY_SIMPLEX



app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
   

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=5)
    return dict(image_np=image_np,rect_points=rect_points, class_names=class_names, class_colors=class_colors)
    # return image_np

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
    #	pass
       print("Converting..")
       file = request.files['file']
       in_memory_file = io.BytesIO()
       file.save(in_memory_file)
       data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
       color_image_flag = 1
       img = cv2.imdecode(data, color_image_flag)

       img = imutils.resize(img,width=400,height=300)
       img = imutils.rotate(img,90)
       
       
       # im = Image.fromarray(img)

       # print("Resize..")
       # im.resize((400,300))

       # print("Rotate..")
       # im = im.rotate(90)


       # img = load_image_into_numpy_array(im)

       print ("Detecting objects..")
       #img = detect_objects(img,sess,detection_graph)

       data = detect_objects(img,sess,detection_graph)
       rec_points = data['rect_points']
       class_names = data['class_names']
       img = data['image_np']
       # class_colors = data['class_colors']

       # width = 300
       # height =400
       sorted_by_second = sorted(zip(rec_points, class_names), key=lambda tup: tup[0]['ymin'], reverse=True)
       # for point, name, color in zip(rec_points, class_names, class_colors):
       #          cv2.rectangle(img, (int(point['xmin'] * width), int(point['ymin'] * height)),
       #                        (int(point['xmax'] * width), int(point['ymax'] * height)), color, 3)
       #          cv2.rectangle(img, (int(point['xmin'] * width), int(point['ymin'] * height)),
       #                        (int(point['xmin'] * width) + len(name[0]) * 6,
       #                         int(point['ymin'] * height) - 10), color, -1, cv2.LINE_AA)
       #          cv2.putText(img, name[0], (int(point['xmin'] * width), int(point['ymin'] * height)), font,
       #                      0.3, (0, 0, 0), 1)

       
       im = Image.fromarray(img)

       reading = ''
       for el in sorted_by_second:
          reading+=el[1][0][0]
       print(reading)
       print("Saving..")
       im.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
       #file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return json.dumps({'filename':file.filename,'reading':reading})
       # extension = os.path.splitext(file.filename)[1]
       # f_name = str(uuid.uuid4()) + extension
       # file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
    #return render_template('index.html')



app.run(host="0.0.0.0", port=5000)
