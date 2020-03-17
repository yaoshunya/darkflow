"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob
#import lie_learn.spaces.S2 as S2
import numpy as np
import pdb
import cv2
import pickle
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
import seaborn as sns


def _pp(l): # pretty printing
    for i in l: print('{}: {}'.format(i,l[i]))

def pascal_voc_clean_xml(ANN, pick, exclusive = False):
    cur_dir = os.getcwd()
    dumps = list()

    with open('data/ann_anchor_data/annotations_only_iou.pickle',mode = 'rb') as f:
        dumps = pickle.load(f)

    return dumps
