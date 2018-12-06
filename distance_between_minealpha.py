from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import math

def sp(image, caption="", seconds=0.0):
    cv2.imshow(caption, image)
    cv2.waitKey(int(seconds*1000))
 

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
    help="path ")
args=vars(ap.parse_args())

image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(blurred, 75,200)

cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = []

if len(cnts)>0:
    cnts = sorted(cnts, key=cv2.contourArea,reverse=True)
    for c in cnts:
        size = cv2.contourArea(c)
        if size>100:
            docCnt.append(c)
        else:
            break
    
    coin=sorted(docCnt, key=lambda docCnt: cv2.boundingRect(docCnt)[0])[0]
    (x,y,w,h) = cv2.boundingRect(coin)
    cv2.drawContours(image, [coin], -1,(0,0,255),2)
    cv2.putText(image,"2cm",(x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2)
    
    for c in docCnt:
        for i in docCnt:
            (x1,y1),radius1=cv2.minEnclosingCircle(c)
            (x2,y2),radius2=cv2.minEnclosingCircle(i)
            if x1==x2 and y1==y2 and radius1==radius2:
                continue
            distance = math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)) /w*2
            
            ti = image.copy()
            cv2.drawContours(ti, [c],-1,(0,255,0),2)
            cv2.drawContours(ti, [i],-1,(0,255,0),2)
            cv2.line(ti,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(ti,"{:.1f}cm".format(distance),(int((x2-x1)/2+x1),int((y2-y1)+y1-20)),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            
            sp(ti,seconds=1)
            

sp(image)