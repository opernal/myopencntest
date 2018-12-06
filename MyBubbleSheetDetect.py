from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

def showpause(image, caption=""):
    '''
    width=500.0
    h, w, d = image.shape
    scale = width/w
    nH,nW = h*scale, w*scale
    newimg = cv2.resize(image, (int(nW), int(nH)))
    '''
    cv2.imshow(caption, image)
    cv2.waitKey(0)
 
 


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

ANSWER_KEY={0:0,1:2,2:1,3:3,4:4,
            5:2,6:2,7:0,8:3,9:4,
            10:1,11:3,12:1,13:3,14:0,
            15:2,16:3,17:1,18:0,19:3,
            20:4,21:1,22:2,23:3,24:0,
            25:0,26:1,27:2,28:3,29:4,
            30:0,31:1,32:2,33:3,34:4,
            35:0,36:1,37:2,38:3,39:4,
            40:0,41:1,42:2,43:3,44:4,
            45:0,46:1,47:2,48:3,49:0,
            }

image = cv2.imread(args["image"])

'''
cv2.imshow("0", image)

image = image/255.0
image = cv2.pow(image, 5)

cv2.imshow("1", image)
cv2.waitKey(0)
'''
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("g",gray)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
#cv2.imshow("b",blurred)
edged = cv2.Canny(blurred, 75, 200)

'''
small = cv2.resize(edged,(0,0), fx=0.5, fy=0.5)
showpause(small)
thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
small = cv2.resize(thresh,(0,0), fx=0.5, fy=0.5)
showpause(small,"2")
#showpause(edged)
#cv2.imshow("",edged)
#cv2.waitKey(0)            :0,:1,:2,:3,:4,
'''

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = []


if len(cnts)>0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        
        if len(approx) == 4:
            docCnt.append(approx)
            print(cv2.contourArea(c))
            #if len(docCnt)==2:
            #    break

print(len(docCnt))
totalScore=0
#cv2.waitKey(0)
for nImage in range(0,2):
    
    paper = four_point_transform(image, docCnt[1-nImage].reshape(4,2))
    warped = four_point_transform(gray, docCnt[1-nImage].reshape(4,2))
    #warped = cv2.GaussianBlur(warped, (5,5), 0)
    
    #small=cv2.resize(warped, (0,0), fx=0.5, fy=0.5)
    #showpause(small)
    
    #cv2.imshow("1",paper)
    #cv2.imshow("2",warped)
    #cv2.waitKey(0)
    #showpause(warped)
    
    thresh = cv2.threshold(warped, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    #thresh = cv2.dilate(thresh, None, iterations=1)
    #thresh = cv2.erode(thresh, None, iterations=1)
    
    h = np.size(thresh,0)
    w = np.size(thresh,1)
    print("{:2d} {:2d}".format(h,w))
    #showpause(thresh)
    #small=cv2.resize(thresh, (0,0), fx=0.5, fy=0.5)
    #showpause(small)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    questionCnts = []
    
    print("counts {:2d}".format(len(cnts)))
     
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        ar = w/ float(h)
        
        if w>=20 and h >=20 and ar>=0.6 and ar<=1.5 and w<50:
            questionCnts.append(c)
            #cv2.drawContours(thresh, [c], -1, (0,0,255), 1)
            print(" w={:3d}% h={:3d}% ar={:.2f}%".format(w,h,ar))
            #showpause(thresh)
        #else:
            #if w>10:
                #cv2.drawContours(thresh, [c], -1, (0,255,0), 1)
                #print(" w={:3d}% h={:3d}% ar={:.2f}%".format(w,h,ar))
                #showpause(thresh)
                
            
            
     
    print("over")
    print(len(questionCnts))
    #showpause(paper)
    
    questionCnts = contours.sort_contours(questionCnts,
        method="top-to-bottom")[0]
    correct=0
    
    for (q,i) in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts = contours.sort_contours(questionCnts[i:i+5])[0]
        bubbled=None
    
        for (j,c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
    
            
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            #showpause(mask)        
            total = cv2.countNonZero(mask)
            
            if bubbled is None or total>bubbled[0]:
                bubbled = (total, j)
        
        
        k = ANSWER_KEY[q+nImage*25]
        
        if k==bubbled[1]:
            color=(0, 255, 0)
            correct += 1
        else:
            color=(0, 0, 255)
            
        (x,y),radius = cv2.minEnclosingCircle(cnts[k])
        center= (int(x),int(y))
        cv2.circle(paper,center,int(radius),color,2)
        #cv2.drawContours(paper, [cnts[k]], -1, color, 3)
        
    totalScore += correct
    if nImage==1:
        score = totalScore/len(ANSWER_KEY)*100
        cv2.putText(paper, "{:.2f}%".format(score), (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    imagereduced = cv2.resize(image, (0,0), fx=0.55, fy=0.55)
    cv2.imshow("Original", imagereduced)
    cv2.moveWindow("Original",400,20)
    paper = cv2.resize(paper, (200,700))
    cv2.imshow("Exam{:2d}".format(nImage), paper)
    cv2.moveWindow("Exam{:2d}".format(nImage),nImage*200,20)

        

cv2.waitKey(0)