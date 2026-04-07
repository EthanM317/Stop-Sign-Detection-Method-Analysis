import cv2 as cv
import numpy as np
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import time


def find_stop_sign(image):
    """
    Given an image I, returns the bounding box 
    for the detected stop sign using colour detection with image pyramids.
    """
    #hsv allows us to ignore saturation & brightness so we can focus on hue
    I = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    #storing red hsv values for comparison
    #red exists at start and end of scale since hue is circular, so we need 2 ranges
    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])
    
    #masks are a binary filter that will state for every pixel in I if it falls into the red range
    mask1 = cv.inRange(I, lower_red1, upper_red1)
    mask2 = cv.inRange(I, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    #remove noise
    '''
    Morphology eliminates noise in the overall structure. It works similar to applying
    a gaussian filter to get rid of noise. The difference is, a guassian filter would
    blur our masked image, turning the discrete values into continuous values.
    Morphology avoids this by simply eliminating noisy pixels in the structure and
    keeping all the values discrete.
    '''
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv.contourArea(contour)
        #ignore small noise
        if area > 500:
            x,y,w,h = cv.boundingRect(contour)
            return np.array([y, x, h, w]).astype(int)
    