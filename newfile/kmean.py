import os
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import cv2
import pickle
import math
import pdb
from sklearn.cluster import KMeans
import xml.etree.ElementTree as ET
import glob


def _pp(l): # pretty printing
    for i in l: print('{}: {}'.format(i,l[i]))

def pascal_voc_clean_xml(ANN, pick, exclusive = False):
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')
    size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i+1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        # actual parsing
        in_file = open(file)
        tree=ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text)
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()

        for obj in root.iter('object'):
                current = list()
                name = obj.find('name').text
                if name not in pick:
                        continue

                xmlbox = obj.find('bndbox')
                xn = int(float(xmlbox.find('xmin').text))
                xx = int(float(xmlbox.find('xmax').text))
                yn = int(float(xmlbox.find('ymin').text))
                yx = int(float(xmlbox.find('ymax').text))
                current = [name,xx-xn,yx-yn]
                all += [current]

        add = [[jpg, [w, h, all]]]
        dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]]+=1
                else:
                    stat[current[0]] =1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps

if __name__ == '__main__':
    print('start k-means')
    dump = list()
    path = '../data/VOC2012/AnnotationsTrain_1' #残差を計算したい対象
    pick = ['car','Truck']
    annotations = pascal_voc_clean_xml(path,pick)
    dump += annotations
    path = '../data/VOC2012/AnnotationsTrain_2' #残差を計算したい対象
    pick = ['car','Truck']
    annotations = pascal_voc_clean_xml(path,pick)
    dump += annotations
    path = '../data/VOC2012/AnnotationsTrain_3/AnnotationsTrain_1' #残差を計算したい対象
    pick = ['car','Truck']
    annotations = pascal_voc_clean_xml(path,pick)
    dump += annotations
    path = '../data/VOC2012/AnnotationsTrain_4/AnnotationsTrain_2' #残差を計算したい対象
    pick = ['car','Truck']
    annotations = pascal_voc_clean_xml(path,pick)
    dump += annotations
    #pdb.set_trace()

    all = list()
    for i in range(len(dump)):
        for j in range(len(dump[i][1][2])):
            coord = np.array(dump[i][1][2][j])
            x_coord = coord[1]
            y_coord = coord[2]
            XY = (x_coord,y_coord)
            all.append(XY)

    cls  = KMeans(n_clusters = 10)
    pred = cls.fit_predict(all)
    centers = cls.cluster_centers_
    print(centers)
    pdb.set_trace()
    
