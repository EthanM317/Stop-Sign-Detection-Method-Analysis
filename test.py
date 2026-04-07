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
    util.detect_all('data/images', 'COLOUR_MATCHING')
    
if __name__ == '__main__':
    main()