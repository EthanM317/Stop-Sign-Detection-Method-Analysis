import cv2 as cv
import numpy as np
import argparse
import os
import xml.etree.ElementTree as ET
import time
import matplotlib.pyplot as plt
from pathlib import Path
import util

def gen_gaussian_pyramid(I, levels=6):
    G = I.copy()
    gpI = [G]
    for i in range(levels):
        G = cv.pyrDown(G)
        gpI.append(G)
    return gpI

# Finds stop sign with trial and error found optimal threshold of 0.29
def find_stop_sign(T, I, threshold=0.29):
    """
    Given a traffic stop sign template T and an image I, returns the bounding box 
    for the detected stop sign using template matching with image pyramids.
    
    Uses normalized correlation coefficient method TM_CCOEFF_NORMED
        
    Returns [0,0,1,1] if stop sign wasn't found.

    I also found that keeping the template and image in color improved accuracy by about 3%, likely since stop signs are always red
    """
    
    
    # Construct image pyramid for image only, this has improved accuracy compared to constructing for the image, and it also yeilds higher accuracy than constructing for both
    max_dim = max(I.shape[0], I.shape[1])
    n_levels = int(np.log2(max_dim)) - 4  
    n_levels = min(n_levels, 6) 
    
    pyramid = gen_gaussian_pyramid(I, levels=n_levels)
    
    best_val = -1
    best_loc = None
    best_scale = 0
    
    th, tw = T.shape[:2]
    
    # Perform template matching at each scale
    for scale_idx, I_scale in enumerate(pyramid):
        #skip if image is smaller than template
        if I_scale.shape[0] < th or I_scale.shape[1] < tw:
            continue
        
      
        R = cv.matchTemplate(I_scale, T, cv.TM_CCOEFF_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(R)
        
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale_idx
    
    if best_val < threshold:
        return np.array([0, 0, 1, 1]).astype(int)
    
    # Scale it back
    scale_factor = 2 ** best_scale
    
    actual_loc = np.array(best_loc) + np.array([tw//2, th//2])
    
    original_loc = actual_loc * scale_factor
    
    # Calculate bounding box 
    top = int(original_loc[1] - (th * scale_factor) // 2)
    left = int(original_loc[0] - (tw * scale_factor) // 2)
    height = int(th * scale_factor)
    width = int(tw * scale_factor)
    
    return np.array([top, left, height, width]).astype(int)


def draw_rect(I, bbox):
    I_ = np.copy(I)
    c = (1.0, 0, 0) if I_.dtype == 'float32' else (255, 0, 0)
    cv.rectangle(I_, bbox, c, 4)
    return I_


def show_image(image_path, template_path):

    I = cv.imread(image_path)
    if I is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)
    
    # Parse XML
    xml_path = image_path.replace('.png', '.xml').replace('images', 'annotations')
    objects = util.parse_xml(xml_path)
    
    # Draw bounding boxes
    I_display = I.copy()
    for obj in objects:
        xmin, ymin = obj['xmin'], obj['ymin']
        xmax, ymax = obj['xmax'], obj['ymax']
        
        cv.rectangle(I_display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        label = obj['name']
        cv.putText(I_display, label, (xmin, ymin-10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(I_display)
    plt.title(f"Image: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def detect_image(image_path, template_path):

    T = cv.imread(template_path)
    if T is None:
        print(f"Error: Could not load template {template_path}")
        return
    
    I = cv.imread(image_path)
    if I is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    I_rgb = cv.cvtColor(I, cv.COLOR_BGR2RGB)
    
    bbox = find_stop_sign_template_matching(T, I)
    
    # Check if stop sign was detected
    detected = not (bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 1 and bbox[3] == 1)
    
    # Parse ground truth
    xml_path = image_path.replace('.png', '.xml').replace('images', 'annotations')
    has_stop = util.has_stop_sign(xml_path)
    
    # Draw detection
    I_display = I_rgb.copy()
    if detected:
        I_display = draw_rect(I_display, bbox)
    
    # Add detection status text
    status = "STOP SIGN DETECTED" if detected else "NO STOP SIGN DETECTED"
    color = (0, 255, 0) if (detected and has_stop) or (not detected and not has_stop) else (255, 0, 0)
    cv.putText(I_display, status, (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(I_display)
    plt.title(f"Detection Result: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Ground Truth: {'Yes' if has_stop else 'No'}")
    print(f"Detected: {'Yes' if detected else 'No'}")


