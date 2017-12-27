
# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import matplotlib.pyplot as plt
from digits import *
import numpy as np
import sys
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
#import sys

#sys.path.append('/usr/local/lib/python2.7/site-packages')
 
# define the dictionary of digit segments so we can identify
# each digit on the thermostat

classifier_fn = "digits_svm.dat"

model = cv2.ml.SVM_load(classifier_fn)




blur_block_size = 25 # odd
threshold_block_size = 31 # odd
threshold_constant = 3
threshold = 110 # region average pixel value segment detection limit
morph_block_size = 8
number_of_digits = 5


def process_image(SSD):
    """Process SSD image."""
    gray = cv2.cvtColor(SSD, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_block_size, blur_block_size), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 
          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
          threshold_block_size, threshold_constant)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,
          np.ones((morph_block_size, morph_block_size), np.uint8))
    return thresh


# load the example image
image = cv2.imread(sys.argv[1])
 
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, width=500 ,height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 100 , 255)


# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None


#rec_contours = []
# loop over the contours
rec_contours = []
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)
 
	# if the contour has four vertices, then we have found
	# the thermostat display

	if len(approx) == 4:
		rec_contours.append(approx)
		#break

		
		

# extract the thermostat display, apply a perspective transform
# to it

#for rec in rec_contours:

#ret,thresh = cv2.threshold(gray,43,255,cv2.THRESH_BINARY_INV)
#plt.imshow(thresh)
#plt.show()

#thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 48, 7)
#plt.imshow(thresh)
#plt.show()
print(len(rec_contours))
for rec in rec_contours:
  warped = four_point_transform(gray, rec_contours[0].reshape(4, 2))
  output = four_point_transform(image, rec_contours[0].reshape(4, 2))

  plt.imshow(warped)
  plt.show()
  gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (blur_block_size, blur_block_size), 0)
  thresh = cv2.adaptiveThreshold(blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            threshold_block_size, threshold_constant)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,
            np.ones((morph_block_size, morph_block_size), np.uint8))
  #thresh = imutils.resize(thresh, height=500)
  # plt.imshow(thresh)
  # plt.show()

# plt.imshow(thresh)
# plt.show()
#  find contours in the thresholded image, then initialize the
# digit contours lists
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# digitCnts = []
# clf = joblib.load("digits_cls.pkl")
# ctrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# ctrs = ctrs[0] if imutils.is_cv2() else ctrs[1]
# print(len(ctrs))
# rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# # For each rectangular region, calculate HOG features and predict
# # the digit using Linear SVM.
# for rect in rects:
#     # Draw the rectangles
#     #cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
#     # Make the rectangular region around the digit
#     leng = int(rect[3] * 1.6)
#     pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
#     pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
#     roi = thresh[pt1:pt1+leng, pt2:pt2+leng]
#     plt.imshow(roi)
#     plt.show()
    # Resize the image
    # roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    # roi = cv2.dilate(roi, (3, 3))
    # # Calculate the HOG features
    # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    #nbr = model.predict(roi_hog_fd)
    #print(nbr[0])
    #cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)


# # loop over the digit area candidates
# for c in cnts:
# 	# compute the bounding box of the contour 
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	print("%d %d"%(w,h))
# 	print(len(cnts))
 
# 	# if the contour is sufficiently large, it must be a digit
# 	#if w >= 200 and h >=200:
# 	digitCnts.append(c)
# 	print("hello")
# 	cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
# 	bin_roi = thresh[y:y+h,x:x+w]
# 	m = bin_roi != 0
# 	#if not 0.1 < m.mean() < 0.4:
# 	#	print("hello")
# 	#	continue
# 	s = 1.5*float(h)/SZ
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
# 	digit = model.predict(sample)[1][0]
# 	print("THis digits is a %d " %digit)
# 	#roismall = cv2.resize(roi,(10,10))
# 	plt.imshow(bin_roi)
# 	plt.show()
	#roismall = roismall.reshape((1,100))
	#roismall = np.float32(roismall)




