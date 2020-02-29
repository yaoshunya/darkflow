import os
import numpy as np
import pickle
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

if __name__ == "__main__":
    with open("../data/out_data/iou_label_conf_01.pickle","rb") as f:
        iou_label_01 = pickle.load(f)
    with open("../data/out_data/precision_conf_01.pickle","rb") as f:
        precision_01 = pickle.load(f)
    with open("../data/out_data/recall_conf_01.pickle","rb") as f:
        recall_01 = pickle.load(f)
    with open("../data/out_data/R_conf_05.pickle","rb") as f:
        R = pickle.load(f)
    with open("../data/out_data/T_0_conf_05.pickle","rb") as f:
        T_0 = pickle.load(f)
    with open("../data/out_data/T_1_conf_05.pickle","rb") as f:
        T_1 = pickle.load(f)

    with open("../data/out_data/precision_conf_02.pickle","rb") as f:
        precision_02 = pickle.load(f)
    with open("../data/out_data/recall_conf_02.pickle","rb") as f:
        recall_02 = pickle.load(f)
    
    with open("../data/out_data/precision_conf_03.pickle","rb") as f:
        precision_03 = pickle.load(f)
    with open("../data/out_data/recall_conf_03.pickle","rb") as f:
        recall_03 = pickle.load(f)
    with open('../data/out_data/precision_conf_04.pickle','rb') as f:
        precision_04 = pickle.load(f)
    with open('../data/out_data/recall_conf_04.pickle','rb') as f:
        recall_04 = pickle.load(f)
    #pdb.set_trace()

    with open('../data/out_data/precision_conf_05.pickle','rb') as f:
        precision_05 = pickle.load(f)
    with open('../data/out_data/recall_conf_05.pickle','rb') as f:
        recall_05 = pickle.load(f)
    
    with open('../data/out_data/precision_conf_06.pickle','rb') as f:
        precision_06 = pickle.load(f)
    with open('../data/out_data/recall_conf_06.pickle','rb') as f:
        recall_06 = pickle.load(f)
    with open('../data/out_data/precision_conf_07.pickle','rb') as f:
        precision_07 = pickle.load(f)
    with open('../data/out_data/recall_conf_07.pickle','rb') as f:
        recall_07 = pickle.load(f)
    """
    with open('../data/out_data/precision_conf_08.pickle','rb') as f:
        precision_08 = pickle.load(f)
    with open('../data/out_data/recall_conf_08.pickle','rb') as f:
        recall_08 = pickle.load(f)
    with open('../data/out_data/precision_conf_09.pickle','rb') as f:
        precision_09 = pickle.load(f)
    with open('../data/out_data/recall_conf_09.pickle','rb') as f:
        recall_09 = pickle.load(f)
    """
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
    X = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    #pdb.set_trace()
    p_01 = np.mean(np.array(precision_01))
    r_01 = np.mean(np.array(recall_01))
    p_02 = np.mean(np.array(precision_02))
    r_02 = np.mean(np.array(recall_02))
    p_03 = np.mean(np.array(precision_03))
    r_03 = np.mean(np.array(recall_03))
    p_04 = np.mean(np.array(precision_04))
    r_04 = np.mean(np.array(recall_04))
    p_05 = np.mean(np.array(precision_05))
    r_05 = np.mean(np.array(recall_05))
    p_06 = np.mean(np.array(precision_06))
    r_06 = np.mean(np.array(recall_06))
    p_07 = np.mean(np.array(precision_07))
    r_07 = np.mean(np.array(recall_07))
    #p_08 = np.mean(np.array(precision_08))
    #r_08 = np.mean(np.array(recall_08))
    #p_09 = np.mean(np.array(precision_09))
    #r_09 = np.mean(np.array(recall_09))
    plt.plot([r_01,r_02,r_03,r_04,r_05,r_06,r_07],[p_01,p_02,p_03,p_04,p_05,p_06,p_07])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('../../GoogleDrive/precision_recall.png')
    plt.clf()


    #pdb.set_trace()
    
    plt.plot(X,[p_01,p_02,p_03,p_04,p_05,p_06,p_07])
    plt.xlabel('confidence')
    plt.ylabel('precision')
    plt.savefig('../../GoogleDrive/precision.png')
    plt.clf()

    plt.plot(X,[r_01,r_02,r_03,r_04,r_05,r_06,r_07])
    plt.xlabel('confidence')
    plt.ylabel('recall')
    plt.savefig('../../GoogleDrive/recall.png')
    plt.clf()
    
    #print('iou:{0}'.format(iou_mean))
    #print('precision:{0}'.format(precision_mean))
    #print('recall:{0}'.format(recall_mean))
