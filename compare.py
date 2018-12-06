from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import tkinter

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0]*imageA.shape[1])
    return err

def compare_images(imageA, imageB, title):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m,s))
    
    ax = fig.add_subplot(1,2,1)
    plt.imshow(imageA, cmap=plt.get_cmap('gray'))
    plt.axis("off")
    
    ax=fig.add_subplot(1,2,2)
    plt.imshow(imageB, cmap=plt.get_cmap('gray'))
    plt.axis("off")
    
    plt.show()
    
ap = argparse.ArgumentParser()
ap.add_argument("-f","--first", required=True,
    help="")
ap.add_argument("-s","--second", required=True,
    help="")

args=vars(ap.parse_args())

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


compare_images(grayA, grayB,'aaa')