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
    """
    dumps_1 = load_data('data/redidual_1',pick,exclusive)
    
    os.chdir(cur_dir)
    dumps_2 = load_data('data/redidual_2',pick,exclusive)
    os.chdir(cur_dir)
    dumps_3 = load_data('data/redidual_3',pick,exclusive)
    #os.chdir(cur_dir)
    #dumps_4 = load_data('data/redidual_4',pick,exclusive)
    
    dumps += dumps_1
    dumps += dumps_2
    dumps += dumps_3
    #dumps += dumps_4
    #pdb.set_trace()
    """
    #pdb.set_trace()
    with open('data/ann_anchor_data/annotations_nor.pickle',mode = 'rb') as f:
        dumps = pickle.load(f)
    """
    T_0 = []
    T_1 = []
    for i in range(len(dumps)):
        for j in range(len(dumps[i][1][0])):
            T_0.append(dumps[i][1][0][j][2][0][0])
            T_1.append(dumps[i][1][0][j][2][0][1])
        print(i)
    #pdb.set_trace()
    #sns.set_style("whitegrid")
    T_0 = np.array(T_0)
    sns.distplot(np.array(T_0))
    #plt.plot(np.array(T_0))
    plt.savefig('data/out_test/T_0.png') 
    plt.clf()
    sns.distplot(np.array(T_1))
    #plt.plot(np.array(T_1))
    plt.savefig('data/out_test/T_1.png') 
    plt.clf()
    """
    #pdb.set_trace()
    return dumps
