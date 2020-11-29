import cv2
import numpy as np
import os
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
import math
from final_model import ElectricMeterModel

electric_model = ElectricMeterModel('meter_model', '/saved_model.pb', 
 		'/labelmap.pbtxt')


PATH_TO_TEST_IMAGES_DIR = 'test_analog'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, file) 
                            for file in os.listdir(PATH_TO_TEST_IMAGES_DIR)]



df = pd.read_csv('analog_readings2.csv')
l = df['label'].to_list()
files = df['Name'].to_list()
l = [int(i) for i in l]
print(l)

readings_lst = []
label_readings = []
num_match = 0
exact_match = 0
img_count=0
exception_match = 0
print(files)
for index,img in enumerate(files):
	cv_img = cv2.imread('test_analog/{}'.format(img),1)
	reading_str = ''
	if(l[index]==-1):
		continue
	meter_type, reading_str, id_number, result, exception = electric_model.predict(cv_img)
	img_count += 1
	if reading_str == '':
		reading_str = '0'
	if(abs(l[index]-int(reading_str)) <=1):
		num_match+=1
	else:
		print(img,' ',reading_str)
		if exception == 1:
			exception_match += 1
	
	if(l[index] == int(reading_str)):
		exact_match+=1

	print('{} / {} images'.format(index, len(TEST_IMAGE_PATHS)))

print('Accuracy {}'.format((float(num_match)/img_count)))
print('True Accuracy {}'.format((float(exact_match)/img_count)))
print('Number of False Positive {}'.format(float(exception_match)/(img_count - num_match)))

