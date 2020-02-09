#from app.models import DigitalMeterModel
#from app.models import AnalogMeterModel
import cv2
import numpy as np
import os
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
import math
from final_model import ElectricMeterModel


#matplotlib.use("TkAgg")




# digital_model = DigitalMeterModel('digital_meter_model','/nas_33705.pb',
# 											'/labelmap.pbtxt')

# analog_model = AnalogMeterModel('analog_meter_model', '/analog_faster_rcnn.pb', 
# 		'/analog_label_map.pbtxt')
electric_model = ElectricMeterModel('meter_model', '/frozen_inference_graph.pb', 
 		'/labelmap.pbtxt')


PATH_TO_TEST_IMAGES_DIR = 'test_analog'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, file) 
                            for file in os.listdir(PATH_TO_TEST_IMAGES_DIR)]



df = pd.read_csv('analog_readings2.csv')
# df = df.replace(np.nan, 0)
l = df['label'].to_list()
files = df['Name'].to_list()
l = [int(i) for i in l]
print(l)

readings_lst = []
label_readings = []
num_match = 0
exact_math = 0
img_count=0
exception_match = 0
#print(TEST_IMAGE_PATHS)
print(files)
for index,img in enumerate(files):
	#print(img)
	cv_img = cv2.imread('test_analog/{}'.format(img),1)
	# # #print(cv_img)
	# plt.imshow(cv_img)
	# plt.show()
	# label_reading = input('Enter reading')
	#label_readings.append('')
	reading_str = ''
	# print(img)
	# #readings = digital_model.predict_reading(cv_img)
	if(l[index]==-1):
		continue
	#readings = analog_model.predict_reading(cv_img)
	meter_type, reading_str, id_number, result, exception = electric_model.predict(cv_img)
	#print(readings)
	# for num_dict in readings:
	# 	reading_str+=num_dict['number']
	# print(l[index], reading_str)
	img_count += 1
	if reading_str == '':
		reading_str = '0'
	if(abs(l[index]-int(reading_str)) <=1):
		num_match+=1
	else:
		if exception == 1:
			exception_match += 1
	
	if(l[index] == int(reading_str)):
		exact_match+=1
	#print(num_match)
# 	#readings_lst.append(reading_str)
	print('{} / {} images'.format(index, len(TEST_IMAGE_PATHS)))

# # # lst = [i for i in range(len(file_names))]
# # #print('File names',file_names)
print('Accuracy {}'.format((float(num_match)/img_count)))
print('True Accuracy {}'.format((float(exact_match)/img_count)))
print('Number of False Positive {}'.format(float(exception_match)/(img_count - num_match)))

# print(file_names)
# print(readings_lst)
# df = pd.DataFrame(list(zip(file_names, readings_lst, label_readings)),
#                columns =['Name', 'val', 'label'])
# file_names = os.listdir(PATH_TO_TEST_IMAGES_DIR)
# df = pd.DataFrame(list(zip(file_names, label_readings)),
#                columns =['Name', 'label'])
# df.to_csv('analog_readings.csv')

