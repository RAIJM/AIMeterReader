# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from digits import *
from sklearn.externals import joblib
from skimage.feature import hog
import deepneuralnet as net
import random
import tflearn.datasets.mnist as mnist
from skimage import io
 

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

blur_block_size = 25 # odd
threshold_block_size = 31 # odd
threshold_constant = 3
threshold = 110 # region average pixel value segment detection limit
morph_block_size = 8
number_of_digits = 5

#classifier_fn = "digits_svm.dat"

#model = cv2.ml.SVM_load(classifier_fn)

clf = joblib.load("digits_cls.pkl")



#rand_index = random.randint(0,len(testX) -1)




# load the example image
image = cv2.imread(sys.argv[1])
 
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None
 
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break

warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))

#threshold the warped image, then apply a series of morphological
#operations to cleanup the thresholded image
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


# gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (blur_block_size, blur_block_size), 0)
# thresh = cv2.adaptiveThreshold(blur, 255, 
#           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
#           threshold_block_size, threshold_constant)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,
#           np.ones((morph_block_size, morph_block_size), np.uint8))


plt.imshow(thresh)
plt.show()

height,width = thresh.shape[:2]

width_quart = int(width * 0.2)

numbers = []
for i in range(0,5):
	numbers.append(thresh[int(height*0.1):int(height*0.7),i*width_quart:(i+1)*width_quart])

for num in numbers:
	plt.imshow(num)
	plt.show()




for roi in numbers:
	(roiH, roiW) = roi.shape
	h = roiH
	w = roiW
	(dW, dH) = (int(roiW * 0.5), int(roiH * 0.5))
	dHC = int(roiH * 0.05)
 
	# define the set of 7 segments
	segments = [
		((0, 0), (w, dH)),	# top
		((0, 0), (dW, h // 2)),	# top-left
		((w - dW, 0), (w, h // 2)),	# top-right
		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		((0, h // 2), (dW, h)),	# bottom-left
		((w - dW, h // 2), (w, h)),	# bottom-right
		((0, h - dH), (w, h))	# bottom
	]
	on = [0] * len(segments)
	for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		# extract the segment ROI, count the total number of
		# thresholded pixels in the segment, and then compute
		# the area of the segment
		segROI = roi[yA:yB, xA:xB]
		total = cv2.countNonZero(segROI)
		area = (xB - xA) * (yB - yA)
 
		# if the total number of non-zero pixels is greater than
		# 50% of the area, mark the segment as "on"
		if total / float(area) > 0.5:
			on[i]= 1
 
	# lookup the digit and draw it on the image
	if(DIGITS_LOOKUP.has_key(tuple(on))):
		digit = DIGITS_LOOKUP[tuple(on)]
		print(digit)
	else:
		print(tuple(on))
	# resized = cv2.resize(roi, (28,28), interpolation = cv2.INTER_AREA)
	# plt.imshow(resized)
	# plt.show()
	# x = np.array(resized)
	# # print(x.shape)
	# x = np.reshape(x,(28,28,-1))
	# result = model.predict([x])[0]
	# prediction = result.tolist().index(max(result))
	# print("Predicition",prediction)
	#roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
	# roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
	# roi = cv2.dilate(roi, (3, 3))
	# # Calculate the HOG features
	# roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	# nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
	# print(nbr[0])

# for bin_roi in numbers:
# 	plt.imshow(bin_roi)
# 	plt.show()

# 	m = bin_roi != 0
# 	#if not 0.1 < m.mean() < 0.4:
# 	#	print("hello")
# 	#	continue
# 	s = 1.5*float(height)/SZ
# 	m = cv2.moments(bin_roi)
# 	c1 = np.float32([m['m10'], m['m01']]) / m['m00']
# 	c0 = np.float32([SZ/2, SZ/2])
# 	t = c1 - s*c0
# 	A = np.zeros((2, 3), np.float32)
# 	A[:,:2] = np.eye(2)*s
# 	A[:,2] = t
# 	bin_norm = cv2.warpAffine(bin_roi, A, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
# 	bin_norm = deskew(bin_norm)
# 	#if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
# 	#	frame[y:,x+w:][:SZ, :SZ] = bin_norm[...,np.newaxis]

# 	sample = preprocess_hog([bin_norm])
# 	plt.imshow(bin_norm)
# 	plt.show()
# 	digit = model.predict(sample)[1][0]
# 	print(digit)

# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# digitCnts = []
# print(len(cnts))

# # loop over the digit area candidates
# for c in cnts:
# 	# compute the bounding box of the contour
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	print((w,h))
 
# 	# if the contour is sufficiently large, it must be a digit
# 	#if w >= 15 and (h >= 30 and h <= 40):
# 	#	digitCnts.append(c)

# # sort the contours from left-to-right, then initialize the
# # actual digits themselves
# cnts = contours.sort_contours(cnts,
# 	method="left-to-right")[0]
# digits = []



# for c in cnts:
# 	# extract the digit ROI
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	roi = thresh[y:y + h, x:x + w]
# 	plt.imshow(roi)
# 	plt.show()