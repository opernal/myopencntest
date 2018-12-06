from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from matplotlib.pyplot import box

def sp(image, caption="", seconds=0.0):
    cv2.imshow(caption, image)
    cv2.waitKey(int(seconds*1000))
    
def midpoint(ptA, ptB):
    return ((ptA[0]+ptB[0])*0.5,(ptA[1]+ptB[1])*0.5)

def drawBox(image, box, scale):
    (tl,tr,br,bl) = box
    (tlblX, tlblY) = midpoint(tl,bl)
    (trbrX, trbrY)= midpoint(tr,br)
    
    D1 = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    D2 = dist.euclidean((tltrX, tltrY),(blbrX, blbrY))
    
    cv2.line(image, (tl[0],tl[1]), (tr[0],tr[1]), (0,255,0), 1)
    cv2.line(image, (bl[0],bl[1]), (br[0],br[1]), (0,255,0), 1)
    cv2.line(image, (tl[0],tl[1]), (bl[0],bl[1]), (0,0,255), 1)
    cv2.line(image, (tr[0],tr[1]), (br[0],br[1]), (0,0,255), 1)
    
    cv2.line(image, ((int)(tlblX), (int)(tlblY)), ((int)(trbrX), (int)(trbrY)), (255,255,0), 2)
    cv2.circle(image, ((int)(trbrX), (int)(trbrY)),3,(255,255,0), -1)
    cv2.circle(image, ((int)(tlblX), (int)(tlblY)),3,(255,255,0), -1)

    cv2.line(image, ((int)(tltrX), (int)(tltrY)), ((int)(blbrX), (int)(blbrY)), (0,255,255), 2)
    cv2.circle(image, ((int)(tltrX), (int)(tltrY)),3,(0,255,255), -1)
    cv2.circle(image, ((int)(blbrX), (int)(blbrY)),3,(0,255,255), -1)
    
    cv2.putText(image, "{:.2f} cm".format(D1/scale), ((int)(trbrX+5), (int)(trbrY)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
    cv2.putText(image, "{:.2f} cm".format(D2/scale), ((int)(tltrX), (int)(tltrY-10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    
    


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
    help="image path ")
ap.add_argument("-w","--width",type=float, required=True,
    help="width of the left-most object in the image (in inches ")
args=vars(ap.parse_args())


image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7,7), 0)
edged = cv2.Canny(blurred, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


(cnts, _) = contours.sort_contours(cnts)
refObj = None

for c in cnts:
    if cv2.contourArea(c)<100:
        continue
    
    box =cv2.minAreaRect(c)
    box =cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    
    box = perspective.order_points(box)
    cX = np.average(box[:,0])
    cY = np.average(box[:,1])
    
    if refObj is None:
        (tl,tr,br,bl) = box
        (tlblX, tlblY) = midpoint(tl,bl)
        (trbrX, trbrY)= midpoint(tr,br)
        
        D1 = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        D2 = dist.euclidean((tltrX, tltrY),(blbrX, blbrY))
        
        recSize = D1*D2
        D = D1 if D1<D2 else D2

        scale = D/args["width"]
        refObj = (box, (cX, cY), scale)
        
        cp = image.copy()
        drawBox(cp, box, scale)
        sp(cp, seconds=2)
        
        continue
    
    cp = image.copy()
    drawBox(cp, box, refObj[2])
    sp(cp, seconds=2)
