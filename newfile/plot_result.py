import os
import numpy as np
import pickle
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

if __name__ == "__main__":
    with open("../data/out_data/iou_label_01.pickle","rb") as f:
        iou_label = pickle.load(f)
    with open("../data/out_data/precision_01.pickle","rb") as f:
        precision = pickle.load(f)
    with open("../data/out_data/recall_01.pickle","rb") as f:
        recall = pickle.load(f)
    with open("../data/out_data/R.pickle","rb") as f:
        R = pickle.load(f)
    with open("../data/out_data/T_0.pickle","rb") as f:
        T_0 = pickle.load(f)
    with open("../data/out_data/T_1.pickle","rb") as f:
        T_1 = pickle.load(f)
    with open("../data/out_data/precision_05.pickle","rb") as f:
        precision_045 = pickle.load(f)
    with open("../data/out_data/recall_05.pickle","rb") as f:
        recall_045 = pickle.load(f)
    with open("../data/out_data/precision_03.pickle","rb") as f:
        precision_03 = pickle.load(f)
    with open("../data/out_data/recall_03.pickle","rb") as f:
        recall_03 = pickle.load(f)

    #pdb.set_trace()

    iou_mean = list()

    plt.xlabel('label')
    plt.ylabel('IoU')
    for i in range(6):
        x = [i+1 for j in range(len(iou_label[i]))]
        plt.scatter(x,iou_label[i],color = 'blue')
        iou_mean.append(np.mean(iou_label[i]))
        plt.scatter(i+1,iou_mean[i],color = 'red')
    plt.savefig('../data/iou_label.png')
    plt.clf()
    """
    plt.xlabel('label')
    plt.ylabel('Precision')
    for i in range(6):
        x = [i+1 for j in range(len(precision_label[i]))]
        plt.scatter(x,precision_label[i],color = 'blue')
        precision_mean.append(np.mean(precision_label[i]))
        plt.scatter(i+1,precision_mean[i],color = 'red')
    plt.savefig('../../GoogleDrive/precision_label.png')
    plt.clf()

    plt.xlabel('label')
    plt.ylabel('Recall')
    for i in range(6):
        x = [i+1 for j in range(len(recall_label[i]))]
        plt.scatter(x,recall_label[i],color = 'blue')
        recall_mean.append(np.mean(recall_label[i]))
        plt.scatter(i+1,recall_mean[i],color = 'red')
    plt.savefig('../../GoogleDrive/recall_label.png')
    plt.clf()
    """
    X = [0.1,0.3,0.5]
    #pdb.set_trace()
    precision_mean = np.mean(np.array(precision))
    recall_mean = np.mean(np.array(recall))
    precision_045_mean = np.mean(np.array(precision_045))
    recall_045_mean = np.mean(np.array(recall_045))
    precision_03_mean = np.mean(np.array(precision_03))
    recall_03_mean = np.mean(np.array(recall_03))

    #pdb.set_trace()
    plt.plot(X,[precision_mean,precision_03_mean,precision_045_mean])
    plt.xlabel('IoU')
    plt.ylabel('precision')
    plt.savefig('../data/precision.png')
    plt.clf()

    plt.plot(X,[recall_mean,recall_03_mean,recall_045_mean])
    plt.xlabel('IoU')
    plt.ylabel('recall')
    plt.savefig('../data/recall.png')
    plt.clf()

    print('iou:{0}'.format(iou_mean))
    print('precision:{0}'.format(precision_mean))
    print('recall:{0}'.format(recall_mean))
