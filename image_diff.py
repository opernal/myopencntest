from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt


ap=argparse.ArgumentParser()
ap.add_argument("-f","--first",required=True,
    help="first input image")
ap.add_argument("-s","--second",required=True,
    help="second")
args=vars(ap.parse_args())

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.rectangle(imageB, (x,y), (x+w,y+h), (0,0,255), 2)
    
#cv2.imshow("Original", imageA)
#cv2.imshow("Modified", imageB)
#cv2.imshow("grayA",grayA)
#cv2.imshow("grayB",grayB)
#cv2.imshow("Diff", diff)
#cv2.imshow("Thresh", thresh)

fig = plt.figure("")

ax = fig.add_subplot(1,2,1)
plt.imshow(imageA)
plt.axis("off")

ax=fig.add_subplot(1,2,2)
plt.imshow(imageB)
plt.axis("off")

plt.show()

cv2.waitKey(0)
    