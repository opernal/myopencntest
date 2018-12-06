import argparse
import cv2

refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    
    global refPt, cropping
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [x, y]
        