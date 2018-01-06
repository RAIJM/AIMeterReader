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
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util



class DigitalMeterModel:

	def __init__(self,model_name,path_to_ckpt,labels_path):
		
		MODEL_NAME = model_name
		PATH_TO_CKPT = MODEL_NAME + path_to_ckpt

		# List of the strings that is used to add correct label for each box.
		PATH_TO_LABELS = os.path.join('data', labels_path)

		NUM_CLASSES = 10

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
	   

	    # Visualization of the results of a detection.
	    vis_util.visualize_boxes_and_labels_on_image_array(
	        image_np,
	        np.squeeze(boxes),
	        np.squeeze(classes).astype(np.int32),
	        np.squeeze(scores),
	        self.category_index,
	        use_normalized_coordinates=True,
	        line_thickness=5)
	    return dict(image_np=image_np,rect_points=rect_points, class_names=class_names, class_colors=class_colors)



	def predict_reading(self,cv_image):

		#resize image
		img = imutils.resize(cv_image,width=400,height=300)
		
		#rotate image due to how model was trained
		img = imutils.rotate(img,90)


		image_np = self.load_cvimage_into_numpy_array(cv_image)

		data = self.detect_objects(img,self.sess,self.detection_graph)

		rec_points = data['rect_points']

		class_names = data['class_names']

		img = data['image_np']

		#sort bounding rects by ymin
		sorted_by_second = sorted(zip(rec_points, class_names), key=lambda tup: tup[0]['ymin'], reverse=True)

		reading = ''
		#append prediction to reading
		for el in sorted_by_second:
			reading+=el[1][0][0]
       		
		return img,reading




class AnalogMeterModel:

	def __init__(self,model_name,path_to_ckpt,labels_path):
		
		MODEL_NAME = model_name
		PATH_TO_CKPT = MODEL_NAME + path_to_ckpt

		# List of the strings that is used to add correct label for each box.
		PATH_TO_LABELS = os.path.join('data', labels_path)

		NUM_CLASSES = 1

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

	

	def load_cvimage_into_numpy_array(self,image):
		im_width = image.shape[1]
		im_height = image.shape[0]
		return image.reshape((im_height,im_width,3)).astype(np.uint8)



	'''
		Args: numpy image, cw/ccw
		Returns: reading between 0-9
		Takes a numpy image and wheter it is cw(even) or ccw(odd) and predicts the corresponding number



	'''
	def read_digit(self,image_np,count):
	    
		height = image_np.shape[1]
		width = image_np.shape[0]

		#convert to grayscale to  be able to threshold
		gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

		#return to upright orientation
		rot = imutils.rotate(gray,90)
	   

		#threshold and open to isloate the pointer
		thresh = cv2.adaptiveThreshold(rot.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	            cv2.THRESH_BINARY_INV,25,10)
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


		#find the resulting countours
		im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)


		#select the largest contour as that should be the pointer
		cnt = max(contours, key = cv2.contourArea)


		#calculate the center of the pointer
		M = cv2.moments(cnt)

		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])


		#find all the white pixels
		cimg = np.zeros_like(thresh)
		cv2.drawContours(cimg, contours, contours.index(cnt), color=255, thickness=-1)
		pts = np.where(cimg == 255)
		arr = zip(pts[0],pts[1])
	   
		#find the furthest white pixel as that should be the tip of the pointer
		max_dist = 0
		max_pixel = None
		for pixel in arr:
			dist = math.sqrt(((cx - pixel[0])**2) + ((cy - pixel[1])**2))
			if dist > max_dist:
				max_pixel = pixel
				max_dist = dist
	   
		tx = max_pixel[1]
		ty = max_pixel[0]
		

		#calculate angle made by center and the tip of the pointer
		theta = math.atan2((ty-cy),(tx-cx))
		phi = math.asin((cy - ty) / math.sqrt(((cy - ty)**2) + ((cx - tx) **2)))

		degrees = math.degrees(theta)

		#take into account 90 degree offset
		if degrees >90 and degrees <=180:
			angle = (degrees -180) - 90
		else:
			angle = degrees + 90


		#convert to from 180/-180 to 0-360
		angle = angle % 360


		#convert corresponding angle to number based on cw/ccw
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

		return reading




	def detect_objects(self,image_np,sess,detection_graph):
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
	        category_index=self.category_index,
	        min_score_thresh=.5
	    )

	    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)



	def predict_reading(self,cv_image):

		#resize image
		image = imutils.resize(cv_image,width=400,height=300)

		#model trained on ccw images
		image = imutils.rotate(image,-90)

		image_np = self.load_cvimage_into_numpy_array(image)

		#detect position of dials
		return_dict = self.detect_objects(image_np,self.sess,self.detection_graph)
		rects = return_dict['rect_points']

		#sort detected dials by position
		rects = sorted(rects, key=lambda rect: rect['ymin'])
		reading=''
		count = 0
		
		for rect in rects:

			#crop image corresponding to bounding rect
			img = image_np[int(rect['ymin']*image.shape[0]):int(rect['ymax']*image.shape[0]),int(rect['xmin']*image.shape[1]):int(rect['xmax']*image.shape[1]),:]
			
			try:
				#predict digit based on cropped image of dial
				digit = self.read_digit(img,count)
			except:
				print("Error")
				digit = -1
			#append prediction to final reading
			reading += str(digit)
			count+=1   #switch between cw and ccw
    	
		return reading


class MeterClassModel:


	def __init__(self,model_name,path_to_ckpt):
		MODEL_NAME = model_name
		PATH_TO_CKPT = MODEL_NAME + path_to_ckpt


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


	def load_image_into_numpy_array(self,image):
		(im_width, im_height) = image.size
	  	return np.array(image.getdata()).reshape(
	    	(im_height, im_width, 1)).astype(np.uint8)

	def predict(self,image):

		#convert to grayscale for prediction model
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		#converting PIL image from opencv image
		im = Image.fromarray(gray)

		#resize to fit to model
		im = im.resize((96,96))
		im = im.rotate(90)

		image_np = self.load_image_into_numpy_array(im)
		image_np_expanded = np.expand_dims(image_np, axis=0)

		#get the input tensor
		x_tensor = self.detection_graph.get_tensor_by_name('Reshape:0')

		#get the output tensor
		output = self.detection_graph.get_tensor_by_name('output/dense/BiasAdd:0')

		#get output
		out  = self.sess.run([output],feed_dict={x_tensor:image_np_expanded})


		pred = np.argmax(out[0],axis=1)

		if (pred[0] == 1):
			return 'digital'
		else:
			return 'analog'




















