import cv2 as cv
import numpy as np
import argparse
import os
import xml.etree.ElementTree as ET
import time
from pathlib import Path
import util
import template_matching

        
def main():
    parser = argparse.ArgumentParser(description='Stop Sign Detection System')
    parser.add_argument('--template-match', help='Detect image with template matching', action='store_true')
    parser.add_argument('--colour-match', help='Detect image with colour matching', action='store_true')
    
    args = parser.parse_args()
    
    if args.template_match:
        util.detect_all('data/images','TEMPLATE_MATCH')
    elif args.colour_match:
        util.detect_all('data/images','COLOUR_MATCH')
    
if __name__ == '__main__':
    main()