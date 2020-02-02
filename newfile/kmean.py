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

if __name__ == '__main__':
    print('start k-means')
    """
    with open('../data/ann_anchor_data/ann_coords_1.pickle',mode = 'rb') as f:
        ann_1 = pickle.load(f)
    with open('../data/ann_anchor_data/ann_coords_2.pickle',mode = 'rb') as f:
        ann_2 = pickle.load(f)
    with open('../data/ann_anchor_data/ann_coords_3.pickle',mode = 'rb') as f:
        ann_3 = pickle.load(f)
    with open('../data/ann_anchor_data/ann_coords_4.pickle',mode = 'rb') as f:
        ann_4 = pickle.load(f)

    dump = list()
    dump += ann_1
    dump += ann_2
    dump += ann_3
    dump += ann_4
    all = list()
    for i in range(len(dump)):
        for j in range(len(dump[i][1])):
            coord = np.array(dump[i][1][j][1])
            x_coord = np.unique(coord[0]).shape[0]
            y_coord = np.unique(coord[1]).shape[0]
            XY = (x_coord,y_coord)
            all.append(XY)

    cls  = KMeans(n_clusters = 5)
    pred = cls.fit_predict(all)
    centers = cls.cluster_centers_
    print(centers)
    pdb.set_trace()
    """

