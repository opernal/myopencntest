import numpy as np
import argparse
import imutils
import cv2

def get_imageROI(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray",gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    #cv2.imshow("blurred",gray)
    edged = cv2.Canny(gray, 20, 100)
    #cv2.imshow("before dilate", edged)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)
    #cv2.imshow("edged",edged)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    #print(len(cnts))
    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask,[c], -1, 255, -1)
        
        (x, y, w, h)=cv2.boundingRect(c)
        imageROI = image[y:y+h, x:x+w]
        maskROI = mask[y:y+h, x:x+w]
        imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)
        
        #cv2.imshow("mask", mask)
        #cv2.imshow("maskROI",maskROI)
        return imageROI, maskROI

def add_black_contourline(image):
    border =2 
    (h, w)=image.shape[:2]
    BLACK=[0,0,0]
    constant = cv2.copyMakeBorder(image, border, border, border, border,
        cv2.BORDER_CONSTANT, value=BLACK)
    
    return constant


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
    help="path")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

#imageROI = get_imageROI(image)
(imageROI, maskROI) = get_imageROI(image)
imageROI = add_black_contourline(imageROI)
cv2.imshow("ori",image)
cv2.imshow("",imageROI)

while True:

    (minH,minW)=imageROI.shape[:2]
    minAngle=0
    
    for angle in np.arange(0, 360, 1):
        rotated = imutils.rotate_bound(imageROI, angle)
        #rotated_mask = imutils.rotate_bound(maskROI, angle)
        #rotatedROI = cv2.bitwise_and(rotated, rotated, mask=rotated_mask)
        #rotatedROI = cv2.bitwise_and(rotated, rotated, mask=rotated_mask)
        #cv2.imshow("Rotated (Correct)", rotated)
        #cv2.imshow("Rotated mask(Correct)", rotated_mask)
        #rotatedROI = get_imageROI(rotated)
        #cv2.imshow("Rotated trimed",rotatedROI)
        
        (image1, image2)= get_imageROI(rotated)
        cv2.imshow("aaaa",image1)
        cv2.waitKey(30)

        # Get minimal picture size
        (tH, tW) = image1.shape[:2]
        if tH*tW<minH*minW:
            minH=tH
            minW=tW
            minAngle = angle
     
    #display minimal angle
    rotated = imutils.rotate_bound(imageROI, minAngle)
    (image1, image2)= get_imageROI(rotated)   
    cv2.putText(image1, "Min Shape, W=%d H=%d Angle=%d" % (minW, minH, minAngle),
                (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)     
    cv2.imshow("Min Shape, W=%d H=%d Angle=%d" % (minW, minH, minAngle), image1)
        
#cv2.imshow("",edged)
cv2.waitKey(0)