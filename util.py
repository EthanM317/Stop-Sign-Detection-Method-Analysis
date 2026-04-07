import cv2 as cv
import numpy as np
import argparse
import os
import xml.etree.ElementTree as ET
import time
from pathlib import Path
import util
import template_matching

#Parses xml file
def parse_xml(xml_path):

    if not os.path.exists(xml_path):
        return []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = []
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None:
            name_elem = obj.find('n') 
        
        if name_elem is not None:
            name = name_elem.text
            bndbox = obj.find('bndbox')
            
            if bndbox is not None:
                bbox = {
                    'name': name,
                    'xmin': int(bndbox.find('xmin').text),
                    'ymin': int(bndbox.find('ymin').text),
                    'xmax': int(bndbox.find('xmax').text),
                    'ymax': int(bndbox.find('ymax').text)
                }
                objects.append(bbox)
    
    return objects

# Checks xml file to see if image has stop sign
def has_stop_sign(xml_path):
    objects = parse_xml(xml_path)
    for obj in objects:
        if obj['name'].lower() == 'stop':
            return True
    return False

def detect_all(folder_path, flag):

    template_path = 'template.png'
    T = cv.imread(template_path)
    if T is None:
        print(f"Error: Could not load template {template_path}")
        return
    
    # Grabs all pngs
    png_files = sorted(Path(folder_path).glob('*.png'))
    
    if len(png_files) == 0:
        print(f"No PNG files found in {folder_path}")
        return
    
    total_images = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    
    start_time = time.time()
    
    print(f"{'Filename':<20}\t{'Stop Sign Detected':<20}\t{'Ground Truth':<15}")
    print("-" * 60)
    
    for png_file in png_files:
        total_images += 1
        
        I = cv.imread(str(png_file))
        if I is None:
            print(f"Error loading {png_file}")
            continue
        
        # Detect stop sign
        if flag == 'TEMPLATE_MATCHING':
            bbox = template_matching.find_stop_sign(T, I)
            detected = not (bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 1 and bbox[3] == 1)
        
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