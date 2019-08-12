import jsonpickle
import numpy as np
import cv2
import os
import tensorflow as tf
import zipfile
import sys
import imutils
from collections import defaultdict
from matplotlib import pyplot as plt
from PIL import Image
from utility import *
import math

import matplotlib.pyplot as plt


# sys.path.append("..")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

MODEL_NAME = 'analog_meter_model'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('analog_meter_model', 'analog_label_map.pbtxt')

NUM_CLASSES = 1

# PATH_TO_TEST_IMAGES_DIR = 'test_analog_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 7) ]

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


def load_cvimage_into_numpy_array(image):
    im_width = image.shape[1]
    im_height = image.shape[0]
    return image.reshape((im_height,im_width,3)).astype(np.uint8)



def read_digit(image_np,count,prev_reading):
    height = image_np.shape[1]
    width = image_np.shape[0]
    gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

    plt.imshow(gray)
    plt.show()

    rot = imutils.rotate(gray,-90)

    plt.imshow(rot)
    plt.show()
   
    thresh = cv2.adaptiveThreshold(rot.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,31,13)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # plt.imshow(thresh)
    # plt.show()


    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    cnt = max(contours, key = cv2.contourArea)
    #cnt = contours[0]
    M = cv2.moments(cnt)

    cimg = np.zeros_like(thresh)
    cv2.drawContours(cimg, contours, contours.index(cnt), color=255, thickness=-1)
    pts = np.where(cimg == 255)
    plt.imshow(cimg)
    plt.show()
    arr = zip(pts[0],pts[1])
   


    #print M
    # cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])
    cx = width/2
    cy = height /2
    print('center: x %d y %d' %(cx,cy))

    cv2.circle(rot,(cx,cy), 1, (0,0,255), -1)


    max_dist = 0
    max_pixel = None
    for pixel in arr:
        dist = math.sqrt(((cx - pixel[0])**2) + ((cy - pixel[1])**2))
        if dist > max_dist:
            max_pixel = pixel
            max_dist = dist
    print("max pixel: x %d y %d" %(max_pixel[1],max_pixel[0]))

    cv2.circle(rot,(max_pixel[1],max_pixel[0]), 1, (0,0,255), -1)

    plt.imshow(rot)
    plt.show()



    tx = max_pixel[1]
    ty = max_pixel[0]
    theta = math.atan2((ty-cy),(tx-cx))
    phi = math.asin((cy - ty) / math.sqrt(((cy - ty)**2) + ((cx - tx) **2)))

    degrees = math.degrees(theta)

    if degrees >90 and degrees <=180:
        angle = (degrees -180) - 90
    else:
        angle = degrees + 90





    #print(theta)
    print("degress %f " %(math.degrees(theta)))
    print("actual angle %f" %(angle))
    #print(phi)

    angle = angle % 360

    print("circle angle %f" %(angle))

    angle = float(angle)


    if angle>=0 and angle <= 36:
            if count % 2 == 0:
                reading = 0
            else:
                reading = 9
    elif angle > 36 and angle <= 72:
        if count % 2 == 0:
            reading = 1
        else:
            reading = 8
    elif angle > 72 and angle <=108:
        if count % 2 == 0:
            reading = 2
        else:
            reading = 7
    elif angle > 108 and angle <= 144:
        if count % 2 == 0:
            reading = 3
        else:
            reading = 6
    elif angle > 144 and angle <= 180:
        if count % 2 == 0:
            reading = 4
        else:
            reading = 5
    elif angle > 180 and angle <=216:
        if count % 2 == 0:
            reading = 5
        else:
            reading = 4
    elif angle > 216 and angle <=252:
        if count % 2 == 0:
            reading = 6
        else:
            reading = 3
    elif angle > 252 and angle <= 288:
        if count % 2 == 0:
            reading = 7
        else:
            reading = 2
    
    elif angle > 288 and angle <= 324:
        if count % 2 == 0:
            reading = 8
        else:
            reading = 1

    elif angle > 324 and angle < 360:
        if count % 2 == 0:
            reading = 9
        else:
            reading = 0




    # edge cases where the dial is close to a number
    # must take into account previous reading to adjust for accuracy
    if(prev_reading!=-1):
        if((angle > 350 and angle<360) or angle < 10):
            if (count %2 ==0):
                if(prev_reading >= 5):
                    reading = 9
                else:
                    reading = 0
            else:
                if(prev_reading >=5):
                    reading = 9
                else:
                    reading = 0

        elif(angle > 26 and angle <=46):
            if count %2 ==0:
                if(prev_reading >= 5):
                    reading = 0
                else:
                    reading = 1
            else:
                if(prev_reading >=5):
                    reading = 8
                else:
                    reading = 9
        elif(angle > 62 and angle < 82):
            if count %2 ==0:
                if(prev_reading >= 5):
                    reading = 1
                else:
                    reading = 2
            else:
                if(prev_reading >=5):
                    reading = 7
                else:
                    reading = 8
        elif(angle > 98 and angle < 118):
            if count %2 ==0:
                if(prev_reading >= 5):
                    reading = 2
                else:
                    reading = 3
            else:
                if(prev_reading >=5):
                    reading = 6
                else:
                    reading = 7
        elif(angle > 134 and angle < 154):
            if count %2 ==0:
                if(prev_reading >= 5):
                    reading = 3
                else:
                    reading = 4
            else:
                if(prev_reading >=5):
                    reading = 5
                else:
                    reading = 6
        elif(angle > 170  and angle < 190):
            if count %2 ==0:
                if(prev_reading >= 5):
                    reading = 4
                else:
                    reading = 5
            else:
                if(prev_reading >=5):
                    reading = 4
                else:
                    reading = 5
        elif(angle > 206 and angle < 226):
            if count %2 ==0:
                if(prev_reading >= 5):
                    reading = 5
                else:
                    reading = 6
            else:
                if(prev_reading >= 5):
                    reading = 3
                else:
                    reading = 4
        elif(angle > 242 and angle < 262):
            if count %2 ==0:
                if(prev_reading >= 5):
                    reading = 6
                else:
                    reading = 7
            else:
                if(prev_reading >= 5):
                    reading = 2
                else:
                    reading = 3
        
        elif(angle > 278 and angle < 298):
            if count %2 ==0:
                if(prev_reading >= 5):
                    reading = 7
                else:
                    reading = 8
            else:
                if(prev_reading >=5):
                    reading = 1
                else:
                    reading = 2
        elif(angle > 314 and angle < 334):
            if count %2 ==0:
                if(prev_reading >= 5):
                    reading = 8
                else:
                    reading = 9
            else:
                if(prev_reading >= 5):
                    reading = 0
                else:
                    reading = 1

    print(reading)
    return reading






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

    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )

    # Visualization of the results of a detection.
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=2)
    #plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(image_np)
    # plt.show()
    # return image_np
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


def main():
    
    image = Image.open('analog.jpg')
    image = image.resize((400,300))

    image_np = load_image_into_numpy_array(image)
    return_dict = detect_objects(image_np,sess,detection_graph)
    print(return_dict)
    rects = return_dict['rect_points']
    rects = sorted(rects, key=lambda rect: rect['ymin'])
    reading=''
    count = 0
    plt.imshow(image_np)
    plt.show()
    # digit = -1
    # for rect in rects:
    #     img = image_np[int(rect['ymin']*image.size[1]):int(rect['ymax']*image.size[1]),int(rect['xmin']*image.size[0]):int(rect['xmax']*image.size[0]),:]
    #     digit = str(read_digit(img,count,int(digit)))
    #     reading += digit
    #     count+=1
    #     plt.imshow(img)
    #     plt.show()
    # print(reading[::-1])
    #im = Image.fromarray(image_np)
    #im.save(str(count)+".jpg")
    #count+=1



if __name__ == '__main__':
    main()