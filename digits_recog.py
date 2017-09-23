#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# built-in modules
import os
import sys
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
import imutils
from imutils import contours

# local modules
#import video
#from common import mosaic

from digits import *

def main():
    

    classifier_fn = 'digits_svm.dat'
    if not os.path.exists(classifier_fn):
        print('"%s" not found, run digits.py first' % classifier_fn)
        return

    if True:
        model = cv2.ml.SVM_load(classifier_fn)
    else:
        model = cv2.ml.SVM_create()
        model.load_(classifier_fn) #Known bug: https://github.com/opencv/opencv/issues/4969

   
    frame = cv2.imread(sys.argv[1])
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image = imutils.resize(frame, height=500)
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

    warped = four_point_transform(gray, displayCnt.reshape(4, 2))

    #plt.imshow(warped)
    #plt.show()
    #output = four_point_transform(image, rec.reshape(4, 2))


    bin = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    bin = cv2.medianBlur(bin, 3)
    plt.imshow(bin)
    plt.show()
    _, contours, heirs = cv2.findContours( bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    try:
        heirs = heirs[0]
    except:
        heirs = []

    for cnt, heir in zip(contours, heirs):
        _, _, _, outer_i = heir
        if outer_i >= 0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not (16 <= h <= 64  and w <= 1.2*h):
            continue
        pad = max(h-w, 0)
        x, w = x-pad/2, w+pad
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))

        bin_roi = bin[y:,x:][:h,:w]
        #gray_roi = frame[y:,x:][:h,:w]

        m = bin_roi != 0
        if not 0.1 < m.mean() < 0.4:
            continue
            '''
            v_in, v_out = gray_roi[m], gray_roi[~m]
            if v_out.std() > 10.0:
                continue
            s = "%f, %f" % (abs(v_in.mean() - v_out.mean()), v_out.std())
            cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)
            '''

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
        if x+w+SZ < frame.shape[1] and y+SZ < frame.shape[0]:
            frame[y:,x+w:][:SZ, :SZ] = bin_norm[...,np.newaxis]

        sample = preprocess_hog([bin_norm])
        digit = model.predict(sample)[0]
        print(digit)
        #cv2.putText(frame, '%d'%digit, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)


    #plt.imshow(frame)
    #plt.imshow(bin)

    #plt.show()
        #cv2.imshow('bin', bin)
    #ch = cv2.waitKey(1)
    #if ch == 27:
    #    break

if __name__ == '__main__':
    main()