#from app.models import DigitalMeterModel
from app.models import AnalogMeterModel
import cv2
import numpy as np
import os
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt

#matplotlib.use("TkAgg")




# digital_model = DigitalMeterModel('digital_meter_model','/nas_33705.pb',
# 											'/labelmap.pbtxt')

analog_model = AnalogMeterModel('analog_meter_model', '/analog_faster_rcnn.pb', 
		'/analog_label_map.pbtxt')


PATH_TO_TEST_IMAGES_DIR = 'test_analog'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, file) 
                            for file in os.listdir(PATH_TO_TEST_IMAGES_DIR)]



df = pd.read_csv('analog_readings.csv')
# df = df.replace(np.nan, 0)
l = df['label'].to_list()
l = [int(i) for i in l]
print(l)

readings_lst = []
label_readings = []
num_match = 0
img_count=0
#print(TEST_IMAGE_PATHS)
for index,img in enumerate(TEST_IMAGE_PATHS):
	#print(img)
	cv_img = cv2.imread(img,1)
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
	readings = analog_model.predict_reading(cv_img)
	#print(readings)
	for num_dict in readings:
		reading_str+=num_dict['number']
	print(l[index], reading_str)
	img_count += 1
	# # if reading_str == '':
	# # 	reading_str = '0'
	if(str(int(reading_str))==str(l[index])):
		num_match+=1
	print(num_match)
# 	#readings_lst.append(reading_str)
	print('{} / {} images'.format(index, len(TEST_IMAGE_PATHS)))

# # # lst = [i for i in range(len(file_names))]
# # #print('File names',file_names)
print('Accuracy {}'.format((float(num_match)/img_count)))

# print(file_names)
# print(readings_lst)
# df = pd.DataFrame(list(zip(file_names, readings_lst, label_readings)),
#                columns =['Name', 'val', 'label'])
# file_names = os.listdir(PATH_TO_TEST_IMAGES_DIR)
# df = pd.DataFrame(list(zip(file_names, label_readings)),
#                columns =['Name', 'label'])
# df.to_csv('analog_readings.csv')
