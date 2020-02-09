import tensorflow as tf
import numpy as np
import imutils
import os
import sys
from utility import *
from utils import label_map_util
from utils import visualization_utils as vis_util
from PIL import Image
import cv2
import math
import statistics

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA["xmax"],boxB["xmax"])
	yA = max(boxA["ymax"], boxB["ymax"])
	xB = min(boxA["xmin"], boxB["xmin"])
	yB = min(boxA["ymin"], boxB["ymin"])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA["xmax"] - boxA["xmin"] + 1) * (boxA["ymin"] - boxA["ymax"] + 1)
	boxBArea = (boxB["xmin"] - boxB["xmax"] + 1) * (boxB["ymin"] - boxB["ymax"] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


class ElectricMeterModel:

	def __init__(self,model_name,path_to_ckpt,labels_path):

		MODEL_NAME = model_name
		PATH_TO_CKPT = os.path.join('data', MODEL_NAME + path_to_ckpt)

		print(PATH_TO_CKPT)

		# List of the strings that is used to add correct label for each box.
		PATH_TO_LABELS = os.path.join('data', MODEL_NAME + labels_path)

		print(PATH_TO_LABELS)

		NUM_CLASSES = 40

		# Loading label map
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)


		#create a tensorflow graph from graph definition from file
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		# l = [n.name for n in detection_graph.as_graph_def().node]
		# print(l)
		self.detection_graph = detection_graph
		self.sess = tf.Session(graph=detection_graph)
		

	#Used to convert opencv image to numpy array
	def load_cvimage_into_numpy_array(self,image):
		im_width = image.shape[1]
		im_height = image.shape[0]
		return image.reshape((im_height,im_width,3)).astype(np.uint8)

	def load_image_into_numpy_array(self,image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

	def detect_objects(self,image_np,sess,detection_graph):
		image_np_expanded = np.expand_dims(image_np, axis=0)
		image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

		# Each box represents a part of the image where a particular object was detected.
		boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

	    # Actual detection.
		(boxes, scores, classes, num_detections) = self.sess.run(
	        [boxes, scores, classes, num_detections],
	        feed_dict={image_tensor: image_np_expanded})

		# Visualization of the results of a detection.
		rect_points, class_names, class_colors = draw_boxes_and_labels(
	        boxes=np.squeeze(boxes),
	        classes=np.squeeze(classes).astype(np.int32),
	        scores=np.squeeze(scores),
	        category_index=self.category_index,
	        min_score_thresh=.5
	    )
	   
		return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)

	
	def predict(self,cv_image):
		exception = 1

		image_np = self.load_cvimage_into_numpy_array(cv_image)

		data = self.detect_objects(image_np,self.sess,self.detection_graph)

		rec_points = data['rect_points']

		class_names = data['class_names']

		boxes = list(zip(rec_points, class_names))

		#print(boxes)

		#exctract labels for meter identificaton number
		n_boxes = list(filter(lambda t: 'n' in t[1][0], boxes))


		#extract labels for dials
		dial_boxes = list(filter(lambda t: t not in n_boxes, boxes))

		#print(dial_boxes)

 		#sort bounding rects by ymin
		sorted_n_boxes = sorted(n_boxes, key=lambda tup: tup[0]['xmin'])
		sorted_dial_boxes = sorted(dial_boxes, key=lambda tup: tup[0]['xmin'])

		filtered_dial_boxes, exception = self.filter_iou(sorted_dial_boxes)

		result_dict = {}
		dial_readings = []

		#go through and edit out x labels
		dial_numbers = [el[1][0].split()[0][0:2] for el in filtered_dial_boxes]
		dial_numbers.reverse()
		final_reading = []
		is_digital = False
		for i, d in enumerate(dial_numbers):
			if 'd' in d:
				is_digital = True
			if i==0:
				final_reading.append(int(d[1]))
			elif('x' in d):
				if(int(final_reading[i-1]) >= 5):
					if(int(d[1]) == 0):
						final_reading.append(9)
					else:
						final_reading.append(int(d[1]) - 1)
				else:
					final_reading.append(int(d[1]))
			else:
				final_reading.append(int(d[1]))
		final_reading.reverse()
		meter_type = 'digital' if is_digital else 'analog'

		for j, el in enumerate(filtered_dial_boxes):
			split_prob = el[1][0].split()
			prob = split_prob[1]
			num = split_prob[0]
			obj = {}
			obj['number'] = final_reading[j]
			obj['prob'] = prob
			obj['dimen'] = el[0]
			dial_readings.append(obj)

		result_dict['dial_boxes'] = dial_readings

		id_numbers = [el[1][0].split()[0][1] for el in sorted_n_boxes]
		meter_id_numbers = []
		for k, el in enumerate(sorted_n_boxes):
			split_prob_n = el[1][0].split()
			prob_n = split_prob_n[1]
			num_n = split_prob_n[0]
			obj_n = {}
			obj_n['number'] = id_numbers[k]
			obj_n['prob'] = prob_n
			obj_n['dimen'] = el[0]
			meter_id_numbers.append(obj_n)
		result_dict['id_number'] = meter_id_numbers
		#print(final_reading)
		return meter_type, ''.join([str(f) for f in final_reading]), ''.join(id_numbers), result_dict, exception



	def filter_iou(self, sorted_boxes):
		exception = 1
		#handling intersections and double labels
		iou_lst = []
		el_removed = []
		if(len(sorted_boxes) > 5):
			exception = 2

			iou_lst = [bb_intersection_over_union(sorted_boxes[i][0], 
							sorted_boxes[i+1][0]) for i in range(len(sorted_boxes)-1)]
			
			#find median iou
			med_iou = statistics.median(iou_lst)
			add_idxs = []
			for i in range(len(iou_lst) + 1):
				if i == len(iou_lst): 
					if i not in add_idxs:
						el_removed.append(sorted_boxes[i])
						break
				elif(abs(iou_lst[i] - med_iou)>0.1): 
					prob1 = float(sorted_boxes[i][1][0].split()[1][:2])
					prob2 = float(sorted_boxes[i+1][1][0].split()[1][:2])
					if(prob1 > prob2): #add the box that has the higher probability
						if(i not in add_idxs): #only add boxes that have not already been added
							el_removed.append(sorted_boxes[i])
							add_idxs.append(i)
							add_idxs.append(i+1)
					else:
						if(i+1 not in add_idxs):
							el_removed.append(sorted_boxes[i+1])
							add_idxs.append(i+1)
							add_idxs.append(i)
				else:
					if(i not in add_idxs):
						add_idxs.append(i)
						el_removed.append(sorted_boxes[i])
		else:
			if(len(sorted_boxes) < 5):
				exception = 3
			el_removed = sorted_boxes

		return el_removed, exception


