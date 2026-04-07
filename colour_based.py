import cv2 as cv
import numpy as np
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import time


def detect_all(folder_path):
    #get images
    images = sorted(Path(folder_path))
    
    if len(images) == 0:
        print(f"No images found in {folder_path}")
        return
    
    #values to track
    total_images = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    
    #also want to track time to complete
    start_time = time.time()
    
    print(f"{'Filename':<20}\t{'Stop Sign Detected':<20}\t{'Ground Truth':<15}")
    print("-" * 60)
    
    #perform detection on each image
    for image in images:
        total_images += 1
        
        I = cv.imread(str(image))
        if I is None:
            print(f"Error loading {image}")
            continue
        
        #detect stop sign
        bbox = find_stop_sign(I)
        detected = not (bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 1 and bbox[3] == 1)
        
        #get ground truth
        # Get ground truth
        xml_file = str(png_file).replace('.png', '.xml').replace('images', 'annotations')
        has_stop = has_stop_sign(xml_file)
        
        if detected and has_stop:
            true_positives += 1
        elif detected and not has_stop:
            false_positives += 1
        elif not detected and has_stop:
            false_negatives += 1
        else:
            true_negatives += 1
        
        detected_str = "Yes" if detected else "No"
        ground_truth_str = "Yes" if has_stop else "No"
        print(f"{png_file.name:<20}\t{detected_str:<20}\t{ground_truth_str:<15}")
    
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {total_images}")
    print(f"True positives: {true_positives}")
    print(f"True negatives: {true_negatives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Total time taken: {total_time_ms:.2f} milliseconds")
    print(f"Average time per image: {total_time_ms/total_images:.2f} milliseconds")
    
    # Calculate accuracy
    if total_images > 0:
        accuracy = (true_positives + true_negatives) / total_images * 100
        print(f"Accuracy: {accuracy:.2f}%")