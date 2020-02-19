import os
import numpy as np
import pickle
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

if __name__ == "__main__":
    with open("../data/out_data/iou_label_conf_05.pickle","rb") as f:
        iou_label = pickle.load(f)
    with open("../data/out_data/precision_conf_05.pickle","rb") as f:
        precision = pickle.load(f)
    with open("../data/out_data/recall_conf_05.pickle","rb") as f:
        recall = pickle.load(f)
    with open("../data/out_data/R_conf_05.pickle","rb") as f:
        R = pickle.load(f)
    with open("../data/out_data/T_0_conf_05.pickle","rb") as f:
        T_0 = pickle.load(f)
    with open("../data/out_data/T_1_conf_05.pickle","rb") as f:
        T_1 = pickle.load(f)

    with open("../data/out_data/precision_conf_04.pickle","rb") as f:
        precision_04 = pickle.load(f)
    with open("../data/out_data/recall_conf_04.pickle","rb") as f:
        recall_04 = pickle.load(f)
    
    with open("../data/out_data/precision_conf_03.pickle","rb") as f:
        precision_03 = pickle.load(f)
    with open("../data/out_data/recall_conf_03.pickle","rb") as f:
        recall_03 = pickle.load(f)
    with open('../data/out_data/precision_conf_045.pickle','rb') as f:
        precision_045 = pickle.load(f)
    with open('../data/out_data/recall_conf_045.pickle','rb') as f:
        recall_045 = pickle.load(f)
    #pdb.set_trace()

    with open('../data/out_data/precision_conf_035.pickle','rb') as f:
        precision_035 = pickle.load(f)
    with open('../data/out_data/recall_conf_035.pickle','rb') as f:
        recall_035 = pickle.load(f)
    
    with open('../data/out_data/precision_0425.pickle','rb') as f:
        precision_0425 = pickle.load(f)
    with open('../data/out_data/recall_0425.pickle','rb') as f:
        recall_0425 = pickle.load(f)
    
    iou_mean = list()
    """
    plt.xlabel('label')
    plt.ylabel('IoU')
    for i in range(6):
        x = [i+1 for j in range(len(iou_label[i]))]
        plt.scatter(x,iou_label[i],color = 'blue')
        iou_mean.append(np.nanmean(iou_label[i]))
        plt.scatter(i+1,iou_mean[i],color = 'red')
    plt.savefig('../data/iou_label.png')
    plt.clf()
    """
    X = [0.3,0.35,0.4,0.425,0.45,0.5]
    #pdb.set_trace()
    p_05 = np.mean(np.array(precision))
    r_05 = np.mean(np.array(recall))
    p_04 = np.mean(np.array(precision_04))
    r_04 = np.mean(np.array(recall_04))
    p_03 = np.mean(np.array(precision_03))
    r_03 = np.mean(np.array(recall_03))
    p_045 = np.mean(np.array(precision_045))
    r_045 = np.mean(np.array(recall_045))
    p_035 = np.mean(np.array(precision_035))
    r_035 = np.mean(np.array(recall_035))
    p_0425 = np.mean(np.array(precision_0425))
    r_0425 = np.mean(np.array(recall_0425))
    #pdb.set_trace()
    plt.plot([r_03,r_035,r_04,r_0425,r_045,r_05],[p_03,p_035,p_04,p_0425,p_045,p_05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('../../GoogleDrive/precision_recall.png')
    plt.clf()


    #pdb.set_trace()
    
    plt.plot(X,[p_03,p_035,p_04,p_0425,p_045,p_05])
    plt.xlabel('confidence')
    plt.ylabel('precision')
    plt.savefig('../../GoogleDrive/precision.png')
    plt.clf()

    plt.plot(X,[r_03,r_035,r_04,r_0425,r_045,r_05])
    plt.xlabel('confidence')
    plt.ylabel('recall')
    plt.savefig('../../GoogleDrive/recall.png')
    plt.clf()
    
    #print('iou:{0}'.format(iou_mean))
    #print('precision:{0}'.format(precision_mean))
    #print('recall:{0}'.format(recall_mean))
