
# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import matplotlib.pyplot as plt
from digits import *
import numpy as np
import sys
#import sys

#sys.path.append('/usr/local/lib/python2.7/site-packages')
 
# define the dictionary of digit segments so we can identify
# each digit on the thermostat

classifier_fn = "digits_svm.dat"

model = cv2.ml.SVM_load(classifier_fn)


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

# load the example image
image = cv2.imread(sys.argv[1])
 
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
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
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)
 
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break
		

# extract the thermostat display, apply a perspective transform
# to it

#for rec in rec_contours:
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))
#plt.imshow(output)
#plt.show()


# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
'''thresh = cv2.threshold(warped, 5, 5,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

blur = cv2.GaussianBlur(warped,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)'''
thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
thresh = cv2.medianBlur(thresh, 2)

plt.imshow(thresh)
plt.show()
# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []
 
# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour 
	(x, y, w, h) = cv2.boundingRect(c)
	print(len(cnts))
 
	# if the contour is sufficiently large, it must be a digit
	if w >= 15 and (h >= 30 and h <= 50):
		digitCnts.append(c)
		print("hello")
		cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
		bin_roi = thresh[y:y+h,x:x+w]
		m = bin_roi != 0
		#if not 0.1 < m.mean() < 0.4:
		#	print("hello")
		#	continue
		s = 1.5*float(h)/SZ
		m = cv2.moments(bin_roi)
		c1 = np.float32([m['m10'], m['m01']]) / m['m00']
		c0 = np.float32([SZ/2, SZ/2])
		t = c1 - s*c0
		A = np.zeros((2, 3), np.float32)
		A[:,:2] = np.eye(2)*s
		A[:,2] = t
		bin_norm = cv2.warpAffine(bin_roi, A, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
		bin_norm = deskew(bin_norm)
		#if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
		#	frame[y:,x+w:][:SZ, :SZ] = bin_norm[...,np.newaxis]

		sample = preprocess_hog([bin_norm])
		digit = model.predict(sample)[1][0]
		print("THis digits is a %d " %digit)
		#roismall = cv2.resize(roi,(10,10))
		plt.imshow(bin_roi)
		plt.show()
		#roismall = roismall.reshape((1,100))
		#roismall = np.float32(roismall)



'''
#digitCnts = contours.sort_contours(digitCnts,
	#method="left-to-right")[0]
#digits = []
#anym_digits = []

# loop over each of the digits
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh[y:y + h, x:x + w]
 
	# compute the width and height of each of the 7 segments
	# we are going to examine
	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
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
		digits.append(digit)
		cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
		cv2.putText(output, str(digit), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
	else:
		anym_digits.append(tuple(on))'''

#print(digits)
#print(anym_digits)
#cv2.imshow("Show",thresh)
#cv2.waitKey()  
#cv2.destroyAllWindows()
#plt.imshow(output)
#plt.show()

