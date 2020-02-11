import os
import numpy as np
import pickle
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

if __name__ == "__main__":
    with open("../data/out_data/iou_label.pickle","rb") as f:
        iou_label = pickle.load(f)
    with open("../data/out_data/precision_label.pickle","rb") as f:
        precision_label = pickle.load(f)
    with open("../data/out_data/recall_label.pickle","rb") as f:
        recall_label = pickle.load(f)
    with open("../data/out_data/R.pickle","rb") as f:
        R = pickle.load(f)
    with open("../data/out_data/T_0.pickle","rb") as f:
        T_0 = pickle.load(f)
    with open("../data/out_data/T_1.pickle","rb") as f:
        T_1 = pickle.load(f)
    pdb.set_trace()
