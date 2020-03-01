import os
import sys
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pdb
import cv2
import pickle
import xml.etree.ElementTree as ET
import math
import random
import shutil
import pandas as pd
from sklearn.metrics import confusion_matrix
from statistics import mean
#  ICP parameters
EPS = 0.00000001
MAXITER = 100

show_animation = False
def load_data(path):

    cur_dir = os.getcwd()
    os.chdir(path)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.pickle')
    size = len(annotations)

    annotations_ = list()

    for i,file in enumerate(annotations):

        with open(file,mode = 'rb') as f:
            annotations_parts = pickle.load(f)

        annotations_ += annotations_parts
    return annotations_,cur_dir

def ICP_matching(ppoints, cpoints):
    """
    Iterative Closest Point matching
    - input
    ppoints: 2D points in the previous frame
    cpoints: 2D points in the current frame
    - output
    R: Rotation matrix
    T: Translation vector
    """
    H = None  # homogeneous transformation matrix

    dError = 1000.0
    preError = 1000.0
    count = 0
    ppoints = np.array([ppoints[0],ppoints[1]])
    cpoints = np.array([cpoints[0],cpoints[1]])
    while dError >= EPS:
        count += 1

        if show_animation:  # pragma: no cover

            plt.cla()
            plt.plot(ppoints[0, :], ppoints[1, :], ".r")
            plt.plot(cpoints[0, :], cpoints[1, :], ".b")
            plt.plot(0.0, 0.0, "xr")
            plt.axis("equal")
            plt.pause(1.0)
            plt.savefig("../../GoogleDrive/icp_test_{0}.png".format(count))

        inds, error = nearest_neighbor_assosiation(ppoints, cpoints)
        Rt, Tt = SVD_motion_estimation(ppoints[:, inds], cpoints)

        # update current points
        cpoints = (Rt @ cpoints) + Tt[:, np.newaxis]

        H = update_homogeneous_matrix(H, Rt, Tt)

        dError = abs(preError - error)
        preError = error
        #print("Residual:", error)

        if dError <= EPS:
            #print("Converge", error, dError, count)
            break
        elif MAXITER <= count:
            print("Not Converge...", error, dError, count)
            break

    R = np.array(H[0:2, 0:2])
    T = np.array(H[0:2, 2])
    #print(R)
    #print(T)
    try:
        R = math.degrees(math.acos(R[0][0]))
    except:
        R = 0
    print(R)
    print(T)
    return R,T


def update_homogeneous_matrix(Hin, R, T):

    H = np.zeros((3, 3))

    H[0, 0] = R[0, 0]
    H[1, 0] = R[1, 0]
    H[0, 1] = R[0, 1]
    H[1, 1] = R[1, 1]
    H[2, 2] = 1.0

    H[0, 2] = T[0]
    H[1, 2] = T[1]

    if Hin is None:
        return H
    else:
        return Hin @ H


def nearest_neighbor_assosiation(ppoints, cpoints):

    # calc the sum of residual errors
    dcpoints = ppoints - cpoints
    d = np.linalg.norm(dcpoints, axis=0)
    error = sum(d)

    # calc index with nearest neighbor assosiation
    inds = []
    for i in range(cpoints.shape[1]):
        minid = -1
        mind = float("inf")
        for ii in range(ppoints.shape[1]):
            d = np.linalg.norm(ppoints[:, ii] - cpoints[:, i])

            if mind >= d:
                mind = d
                minid = ii

        inds.append(minid)

    return inds, error


def SVD_motion_estimation(ppoints, cpoints):

    pm = np.mean(ppoints, axis=1)
    cm = np.mean(cpoints, axis=1)

    pshift = ppoints - pm[:, np.newaxis]
    cshift = cpoints - cm[:, np.newaxis]

    W = cshift @ pshift.T
    u, s, vh = np.linalg.svd(W)

    R = (u @ vh).T
    t = pm - (R @ cm)

    return R, t


def detect_R_T(ann,anchor,path_num):

    dumps = list()
    path = ['redidual_1_an','redidual_2_an','redidual_3_an','redidual_4_an']
    with open('../data/ann_anchor_data/mask_anchor_k.pickle',mode = 'rb') as f:
        mask_anchor = pickle.load(f)
    mask_anchor = np.reshape(mask_anchor,[361,5,1000,1000])
    #mask_anchor_T = np.transpose(np.reshape(mask_anchor,[361,5,1000,1000]),[0,1,3,2])
    mask__ = np.reshape(mask_anchor,[1805,1000,1000])
    mask_ = list()
    for i in range(1805):
        mask_parts = cv2.resize(mask__[i],(250,250))
        mask_parts[mask_parts>0] = 1
        mask_.append(mask_parts)
    mask_ = np.array(mask_)
    iou_list = list()
    iou_affine_list = list()
    kernel = [[1,1,1],[1,-8,1],[1,1,1]]
    for ann_len in range(len(ann)):

        img_name = ann[ann_len][0]
        all = list()
        print(img_name)
        for ann_0_len in range(len(ann[ann_len][1])):

            error = list()
            iou = list()
            name = ann[ann_len][1][ann_0_len][0]
            current = list()

            annotations_x = np.array(ann[ann_len][1][ann_0_len][1][0])
            annotations_y = np.array(ann[ann_len][1][ann_0_len][1][1])

            x_max = np.max(annotations_x)
            x_min = np.min(annotations_x)
            y_max = np.max(annotations_y)
            y_min = np.min(annotations_y)

            mask_annotation = np.zeros([1000,1000])
            mask_annotation[ann[ann_len][1][ann_0_len][1]] = 1
            mask_annotation = cv2.resize(mask_annotation,(250,250))
            mask_annotation[mask_annotation>0] = 1
            mask_annotation = np.tile(mask_annotation[np.newaxis][np.newaxis],[361,5,1,1])
            or_ = np.logical_or(np.reshape(mask_,[361,5,250,250]),mask_annotation).astype(np.int)
            and_ = np.logical_and(np.reshape(mask_,[361,5,250,250]),mask_annotation).astype(np.int)

            or_ = np.sum(np.sum(np.sum(or_,2),2),1)
            and_ = np.sum(np.sum(np.sum(and_,2),2),1)
            iou = and_/or_

            max_index = np.argmax(iou)
            len_ann = len(annotations_x)
            len_anc = [len(np.where(mask_anchor[max_index][i]>0)[0]) for i in range(5)]
            idx = np.abs(np.array(len_anc)-len_ann).argmin()

            iou = np.sum(np.logical_and(np.reshape(mask_,[361,5,250,250])[max_index][idx],mask_annotation[0][0]))/np.sum(np.logical_or(np.reshape(mask_,[361,5,250,250])[max_index][idx],mask_annotation[0][0]))
            print(ann_0_len)

            R_list = list()
            T_list = list()

            """
            anc = np.where(mask_anchor[max_index][idx]>0)
            #anc = (anc[1],anc[0])
            anchor_len_ = len(anc[0])
            ann_len_ = len(ann[ann_len][1][ann_0_len][1][0])

            my_list_ann = []
            my_list_anchor = []

            for k in range(150):
                x = random.randint(0,ann_len_-1)
                y = random.randint(0,anchor_len_-1)
                my_list_ann.append(x)
                my_list_anchor.append(y)
            ann_stack = np.vstack((ann[ann_len][1][ann_0_len][1][0][my_list_ann],ann[ann_len][1][ann_0_len][1][1][my_list_ann]))
            anchor_stack = np.vstack((anc[0][my_list_anchor],anc[1][my_list_anchor]))

            """
            ann_now = ann[ann_len][1][ann_0_len][1]
            A = np.zeros((1000,1000),dtype = 'uint8')
            A[ann_now] = 255
            ann_now = A
            anchor_now = mask_anchor[max_index][idx]

            dst_ann = np.where(cv2.Laplacian(ann_now,cv2.CV_32F,ksize=3)>0)
            dst_anchor = np.where(cv2.Laplacian(anchor_now,cv2.CV_32F,ksize=3)>0)

            ann_len_ = len(dst_ann[0])
            anchor_len_ = len(dst_anchor[0])

            iou_affine = 0
            count = 0

            while(iou > iou_affine or iou_affine < 0.5):
                my_list_ann = []
                my_list_anchor = []
                if iou_affine > 0.8:
                    break
                count += 1
                if count == 100:
                    break
                for k in range(50):
                    x = random.randint(0,ann_len_-1)
                    y = random.randint(0,anchor_len_-1)
                    my_list_ann.append(x)
                    my_list_anchor.append(y)

                ann_stack = np.vstack((dst_ann[0][my_list_ann],dst_ann[1][my_list_ann]))
                anchor_stack = np.vstack((dst_anchor[0][my_list_anchor],dst_anchor[1][my_list_anchor]))

                R,T  = ICP_matching(ann_stack,anchor_stack)

                img = cv2.imread('../data/VOC2012/sphere_data/{0}'.format(img_name))
                with open('../data/mask_ann/{0}_{1}.pickle'.format(img_name[:6],ann_0_len),mode = 'rb') as f:
                    an = pickle.load(f)

                X = np.zeros((1000,1000))
                X[ann[ann_len][1][ann_0_len][1]] = 255
                an = cv2.resize(an,(1000,1000))*255


                an_ = mask_anchor[max_index][idx]
                affine = cv2.getRotationMatrix2D((0,0),R,1.0)
                affine[0][2] += T[1]
                affine[1][2] += T[0]

                pre = cv2.warpAffine(an_,affine,(1000,1000))

                pre_resize = cv2.resize(pre,(250,250))
                pre_resize[pre_resize>0] = 1

                or_ = np.sum(np.logical_or(pre_resize,mask_annotation[0]))
                and_ = np.sum(np.logical_and(pre_resize,mask_annotation[0]))
                iou_affine = and_/or_

            #iou_list.append(iou)
            #iou_affine_list.append(iou_affine)
            print('iou       :{0}'.format(iou))
            print('affine iou:{0}'.format(iou_affine))

            """
            where_ = np.where(pre)
            pre_1 = np.zeros((1000,1000))
            pre_2 = np.zeros((1000,1000))
            pre_1[where_] = 0
            pre_2[where_] = 200
            pre = pre[np.newaxis]
            pre_1 = pre_1[np.newaxis]
            pre_2 = pre_2[np.newaxis]
            pre = np.append(pre,pre_1,0)
            pre = np.append(pre,pre_2,0)

            pre = np.transpose(pre,[1,2,0])
            X = np.tile(np.transpose(X[np.newaxis],[1,2,0]),[1,1,3])

            prediction = cv2.addWeighted(np.asarray(img,np.float64),0.7,np.asarray(pre,np.float64),0.3,0)
            prediction = cv2.addWeighted(np.asarray(prediction,np.float64),0.6,np.asarray(X,np.float64),0.4,0)
            cv2.imwrite('../../GoogleDrive/messigray_n_{0}_{1}.png'.format(ann_len,ann_0_len),prediction)
            cv2.imwrite('messigray_{0}_{1}.png'.format(ann_len,ann_0_len),prediction)
            """
            """
            not_affine = np.append(an_[np.newaxis],np.zeros((1,1000,1000)),0)
            pre_2 = np.zeros((1000,1000))
            pre_2[np.where(an_>0)]= 200
            pre_2 = pre_2[np.newaxis]
            not_affine = np.transpose(np.append(not_affine,pre_2,0),[1,2,0])
            prediction = cv2.addWeighted(np.asarray(img,np.float64),0.7,np.asarray(not_affine,np.float64),0.3,0)
            prediction = cv2.addWeighted(np.asarray(prediction,np.float64),0.6,np.asarray(X,np.float64),0.4,0)
            cv2.imwrite('../../GoogleDrive/not_affine_{0}_{1}.png'.format(ann_len,ann_0_len),prediction)
            ###############################
            """
            current = [name,R,T,x_min,y_min,x_max,y_min,max_index+361*idx]

            all.append(current)

        add = [[img_name,[all]]]
        dumps += add
        """
        if ann_len == 8:
            print('iou_mean:{0}'.format(np.mean(np.array(iou_list))))
            print('iou_affine_mean:{0}'.format(np.mean(np.array(iou_affine_list))))
            pdb.set_trace()
        """
        print("finish:{0}_{1}".format(ann_len,path_num))
        if ann_len % 50 == 0:
            with open('../data/{0}/redidual_parts_{1}.pickle'.format(path[path_num],ann_len//50),mode = 'wb') as f:
                pickle.dump(dumps,f)
            dumps = list()

    with open('../data/{0}/redidual_1.pickle'.format(path[path_num]),mode = 'wb') as f:
            pickle.dump(dumps,f)

    return 0

if __name__ ==  '__main__':

    if not os.path.exists('../data/redidual_1/redidual_parts_1_.pickle'):
        with open('../data/ann_anchor_data/anchor_coords_k.pickle',mode = 'rb') as f:
            anchor = pickle.load(f)

        with open('../data/ann_anchor_data/ann_coords_3_T.pickle',mode = 'rb') as f:
            ann_1 = pickle.load(f)

        print("start detect the redidual between anchors and annotations")
        ann_1 = detect_R_T(ann_1,anchor,2)

        print("finish 1")
