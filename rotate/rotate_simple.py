import numpy as np
import argparse
import imutils
import cv2
import time

def rotate_bound_mine(image, angle):
    (h, w)=image.shape[:2]
    (cX, cY)=(w//2,h//2)
    
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    nW = int((w*cos)+(h*sin))
    nH = int((w*sin)+(h*cos))
    
    M[0,2] += (nW/2) - cX
    M[1,2] += (nH/2) - cY
    
    return cv2.warpAffine(image, M, (nW, nH))

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required= True,
    help="path to the image file")
args = vars(ap.parse_args())


image = cv2.imread(args["image"])


for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate(image, -angle)
    cv2.imshow("Rotated (problematic)", rotated)
    cv2.waitKey(500)
    
for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate_bound(image, angle)
    cv2.imshow("Rotated (Correct)", rotated)
    cv2.waitKey(500)
               
for angle in np.arange(0, 360, 15):
    rotated = rotate_bound_mine(image, angle)
    cv2.imshow("Rotated (Correct)", rotated)
    cv2.waitKey(500)
                  
              