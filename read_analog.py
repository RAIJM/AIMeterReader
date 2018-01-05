import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import math


width = 300
height = 300

filename = 'analog.jpeg'
org = cv2.imread(filename)
gray = cv2.cvtColor(org,cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray, (9, 9), 2,2)
img = imutils.resize(gray,width=300,height=300)
rot = imutils.rotate(img,-90)



# detect circles in the image
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, param1=100,param2=100)
 
# # ensure at least some circles were found
# if circles is not None:
# 	# convert the (x, y) coordinates and radius of the circles to integers
# 	circles = np.round(circles[0, :]).astype("int")
 
# 	# loop over the (x, y) coordinates and radius of the circles
# 	for (x, y, r) in circles:
# 		# draw the circle in the output image, then draw a rectangle
# 		# corresponding to the center of the circle
# 		cv2.circle(img, (x, y), r, (0, 255, 0), 4)
# 		cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
# edged = cv2.Canny(blurred, 50, 200, 255)

thresh = cv2.threshold(rot.copy(), 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(50,50))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


contours= cv2.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv2.moments(cnt)
#print M
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print('center: x %d y %d' %(cx,cy))

cv2.circle(rot,(cx,cy), 5, (0,0,255), -1)

#print(thresh[cx,cy])

white_pixels = []
for i in range(height):
	for j in range(width):
		if(thresh[j,i] == 255):
			white_pixels.append((j,i))



max_dist = 0
max_pixel = None
for pixel in white_pixels:
	dist = math.sqrt(((cx - pixel[0])**2) + ((cy - pixel[1])**2))
	if dist > max_dist:
		max_pixel = pixel
		max_dist = dist

cv2.circle(rot,(max_pixel[1],max_pixel[0]), 5, (0,0,255), -1)



tx = max_pixel[1]
ty = max_pixel[0]
theta = math.atan((cy - ty)/(cx - tx))
phi = math.asin((cy - ty) / math.sqrt(((cy - ty)**2) + ((cx - tx) **2)))

print(theta)
print("degress %f " %(theta/math.pi * 180))
print(phi)

reading = 0
if(theta <= math.pi/2 and theta > 3*math.pi/10 and phi > 0):
	reading = 0
elif(theta <= 3*math.pi/10 and theta > math.pi/10 and phi > 0):
	reading = 9
elif(theta <= math.pi/10 and theta > -math.pi/10 and phi > 0):
	reading = 8
elif(theta <= -math.pi/10 and theta > -3*math.pi/10 and phi < 0):
	reading = 7
elif(theta <= -3*math.pi/10  and theta > -math.pi/2 and phi < 0):
	reading = 6
elif(theta <= math.pi/2 and theta > 3*math.pi/10 and phi < 0):
	reading = 5
elif(theta <= 3*math.pi/10 and theta > math.pi/10 and phi < 0):
	reading = 4
elif(theta <= math.pi/10 and theta > -math.pi/10 and phi < 0):
	reading = 3
elif(theta <= -math.pi/10 and theta > -3*math.pi/10 and phi > 0):
	reading = 2
elif(theta <= -3 * math.pi/10 and theta > -math.pi/2 and phi > 0):
	reading = 1

# print(reading)

'''gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]'''

'''gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]'''

'''corners = cv2.goodFeaturesToTrack(gray,5,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)'''

# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)

# img=cv2.drawKeypoints(gray,kp)
plt.imshow(rot)
plt.show()
#cv2.imshow('dst',img)
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()