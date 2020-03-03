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

def load_data(path,pick,exclusive):
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    cur_dir = os.getcwd()
    os.chdir(path)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.pickle')
    size = len(annotations)

    annotations_ = list()
    #pdb.set_trace()
    for i,file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i+1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        with open(file,mode = 'rb') as f:
            annotations_parts = pickle.load(f)

        annotations_ += annotations_parts
    return annotations_


def pascal_voc_clean_xml(ANN, pick, exclusive = False):
    cur_dir = os.getcwd()
    dumps = list()

    with open('data/ann_anchor_data/annotations_only_iou.pickle',mode = 'rb') as f:
        dumps = pickle.load(f)

    return dumps
