import tensorflow as tf
import numpy as np
import cv2
import imutils
from PIL import Image



MODEL_NAME = 'meter_model'
PATH_TO_CKPT = MODEL_NAME + '/frozen_meter_model.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

	l = [n.name for n in detection_graph.as_graph_def().node]
	print(l)
	sess = tf.Session(graph=detection_graph)


def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
  	return np.array(image.getdata()).reshape(
    	(im_height, im_width, 1)).astype(np.uint8)

def predict(cv_image):

		gray = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
		im = Image.fromarray(gray)
		im = im.resize((96,96))


		image_np = load_image_into_numpy_array(im)
		image_np_expanded = np.expand_dims(image_np, axis=0)

		x_tensor = detection_graph.get_tensor_by_name('Reshape:0')

		output = detection_graph.get_tensor_by_name('output/dense/BiasAdd:0')

		out =   sess.run([output],feed_dict={x_tensor:image_np_expanded})

		pred = np.argmax(out[0],axis=1)

		if pred[0] == 1:
			return 'digital'
		else:
			return 'analog'


def main():
   
    image = cv2.imread('analog2.jpg')
    
    output = predict(image)
    print(output)


if __name__ == '__main__':
    main()