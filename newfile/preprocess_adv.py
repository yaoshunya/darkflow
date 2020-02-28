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
#  ICP parameters
EPS = 0.00001
MAXITER = 100

show_animation = False
def load_data(path):

    cur_dir = os.getcwd()
    os.chdir(path)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.pickle')
    size = len(annotations)

    annotations_ = list()
    #pdb.set_trace()
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
    ppoints = np.array([ppoints[1],ppoints[0]])
    cpoints = np.array([cpoints[1],cpoints[0]])
    while dError >= EPS:
        count += 1

        if show_animation:  # pragma: no cover

            plt.cla()
            plt.plot(ppoints[0, :], ppoints[1, :], ".r")
            plt.plot(cpoints[0, :], cpoints[1, :], ".b")
            plt.plot(0.0, 0.0, "xr")
            plt.axis("equal")
            plt.pause(1.0)
            plt.savefig("icp_test_{0}.png".format(count))

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
    #pdb.set_trace()
    R = np.array(H[0:2, 0:2])
    T = np.array(H[0:2, 2])
    R = math.asin(R[0][0])

    return R, T


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




def get_projection_grid(b, grid_type="Driscoll-Healy"):
    theta, phi = np.meshgrid(np.arange(2 * b) * np.pi / (2. * b),np.arange(2 * b) * np.pi / b, indexing='ij')
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_


def rand_rotation_matrix(deflection=1.0,randnums=None):
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta,phi,z=(np.pi/2,np.pi,1)

    r_=np.sqrt(z)
    V=(
    np.sin(phi)*r_,
    np.cos(phi)*r_,
    np.sqrt(2.0-z)
    )

    st=np.sin(theta)
    ct=np.cos(theta)

    R=np.array(((ct,st,0),(-st,ct,0),(0,0,1)))

    M=(np.outer(V,V)-np.eye(3)).dot(R)

    return M

def rotate_grid(rot,grid):
    x,y,z=grid
    xyz=np.array((x,y,z))
    x_r,y_r,z_r=np.einsum('ij,jab->iab',rot,xyz)
    return x_r,y_r,z_r

def sample_bilinear(signal, rx, ry):
    #pdb.set_trace()
    if len(signal.shape) > 2:
        signal_dim_x = signal.shape[1]
        signal_dim_y = signal.shape[2]
    else:
        signal_dim_x = signal.shape[0]
        signal_dim_y = signal.shape[1]
    rx *= signal_dim_x
    ry *= signal_dim_y


    ix = rx.astype(int)
    iy = ry.astype(int)

    ix0 = ix - 1
    iy0 = iy - 1
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    return (iy1 - ry) * fx1 + (ry - iy0) * fx2

def project_2d_on_sphere(signal, grid , projection_origin=None):


    if projection_origin is None:
        projection_origin = (0, 0, 2 + 1e-3)

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)

    sample = sample_bilinear(signal, rx, ry)


    sample *= (grid[2] <= 1).astype(np.float64)

    if len(sample.shape) > 2:
        sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
        sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)
        sample = (sample - sample_min) / (sample_max - sample_min)
    else:
        sample_min = sample.min(axis=(0,1))
        sample_max = sample.max(axis=(0,1))
        sample = (sample - sample_min) / (sample_max - sample_min)

    sample *= 255
    sample = sample.astype(np.uint8)

    return sample

def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds
    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample

def project_sphere_on_xy_plane(grid, projection_origin):
    sx, sy, sz = projection_origin
    x, y, z = grid

    z = z.copy() + 1


    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))

    return rx, ry

def mask_anchor(anchor,H):
    img_x = 1000#画像の幅
    img_y = 1000#画像の高さ

    step_x=0
    step_y=0
    S = H
    #anchor = np.reshape(anchor,(5,2))
    anchor_size = anchor.shape[0]
    step_size = int(img_x/S)#グリッドに分割した際のグリッド1つ当たりのピクセル数


    mask = np.array([])

    for i in range(S):
        for t in range(S):
            if t==0:
                step_x = 0
            center_x = int((step_x + (step_x + step_size))/2)#gridの中心座標x
            center_y = int((step_y + (step_y + step_size))/2)#gridの中心座標y
            for l in range(anchor_size):
                #pdb.set_trace()
                w_ = float(x[l].split(",")[0])
                h_ = float(x[l].split(",")[1])

                mask_base = np.zeros((img_x,img_y),dtype=int)

                #side = int(w_*1224/488)
                #ver = int(h_*370/488)
                side = int(h_)/2
                ver = int(w_)/2

                side_min = center_x - side
                side_max = center_x + side
                ver_min = center_y - ver
                ver_max = center_y + ver

                if side_min < 0:
                    side_min = 0
                if side_max > img_x:
                    side_max = img_x
                if ver_min < 0:
                    ver_min = 0
                if ver_max > img_y:
                    ver_max = img_y


                mask_base[ver_min:ver_max,side_min:side_max] = 255#mask anchor(歪みなし)の作成
                #mask_base = np.resize(mask_base,(500,500))
                #-------------------------------------------------
                #アンカーの球面写像
                grid = get_projection_grid(b=500)
                rot = rand_rotation_matrix(deflection=1.0)
                grid = rotate_grid(rot,grid)
                mask_base = project_2d_on_sphere(mask_base,grid)#mask anchorに歪みを持たせる
                #-------------------------------------------------


                resize_mask = mask_base.T
                if l == 0:
                    mask = resize_mask[np.newaxis]
                else:
                    mask = np.append(mask,resize_mask[np.newaxis],axis=0)


            step_x += step_size
            if t == 0:
                mask_ = mask[np.newaxis]
            else:
                mask_ = np.append(mask_,mask[np.newaxis],axis=0)
        print(i)
        step_y += step_size
        if i == 0:
            mask_fi = mask_[np.newaxis]
            #mask_fi_row = mask__row[np.newaxis]
        else:
            mask_fi = np.append(mask_fi,mask_[np.newaxis],axis=0)
            #mask_fi_row = np.append(mask_fi_row,mask__row[np.newaxis],axis=0)

    return mask_fi

def pascal_voc_clean_xml(ANN, pick, exclusive = False):
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')
    size = len(annotations)
    mask = np.array([])

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
        t = 0

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

            #make mask annotation from coordinates
            ###############################################################
            mask_prepare = np.zeros((375,1242),dtype=int)
            mask_prepare[yn:yx,xn:xx]=255
            mask_parts = np.array([])
            grid = get_projection_grid(b=500)
            rot = rand_rotation_matrix(deflection=1.0)
            grid = rotate_grid(rot,grid)
            mask_parts = project_2d_on_sphere(mask_prepare,grid)
            #mask_parts = cv2.resize(mask_parts,(19,19))
            mask_parts = np.reshape(mask_parts,(1000,1000)).T

            mask_ = mask_parts[np.newaxis]
            """
            image_sample = cv2.imread('../sphere_data/{0}'.format(jpg))
            mask_tile = np.transpose(np.tile(mask_,[3,1,1]),[2,1,0])
            image_sample = cv2.addWeighted(image_sample,0.5,mask_tile,0.5,0)
            cv2.imwrite('../../../../GoogleDrive/{0}_{1}.png'.format(jpg,t),image_sample)
            #pdb.set_trace()
            """
            current = [name,mask_]
            all += [current]
            t = t + 1
        add = [[jpg,all]]
        #pdb.set_trace()
        dumps += add
        in_file.close()
        ###################################################################


        ###################################################################
    os.chdir(cur_dir)
    all = list()
    current = list()
    add = list()
    #pdb.set_trace()
    return dumps

def make_coords_from_mask(data,flag):

    if flag == 0:#anchorを座標系に変換する場合
        anchor = data
        anchor = np.reshape(anchor,[anchor.shape[0]*anchor.shape[1],anchor.shape[2],anchor.shape[3],anchor.shape[4]])
        anchor_coords = list()

        #pdb.set_trace()
        for i in range(anchor.shape[0]):
            anc_parts = []
            for j in range(anchor.shape[1]):
                anchor_coords_parts = np.array(np.where(anchor[i][j]>0))
                anc_parts.append(anchor_coords_parts)
            anchor_coords.append(anc_parts)
        return anchor_coords

    if flag == 1:#annotationsを座標系に変換する場合
        ann_coords = []
        annotations = data

        #pdb.set_trace()
        for i in range(len(annotations)):
            #pdb.set_trace()
            jpg = annotations[i][0]
            all = []
            #pdb.set_trace()
            for j in range(len(annotations[i][1])):
                #pdb.set_trace()
                #print(j)
                name = annotations[i][1][j][0]
                current = list()
                ann_coords_parts = np.where(np.reshape(annotations[i][1][j][1],[1000,1000])>0)
                ann_coords_parts = (ann_coords_parts[0],ann_coords_parts[1])
                #pdb.set_trace()
                current = [name,ann_coords_parts]
                #pdb.set_trace()
                all += [current]

            add = [[jpg,all]]
            ann_coords += add

        #pdb.set_trace()

        return ann_coords
    return 0

def detect_R_T(ann,anchor,path_num):

    dumps = list()
    path = ['redidual_1_an','redidual_2_an','redidual_3_an','redidual_4_an']
    with open('../data/ann_anchor_data/mask_anchor_k.pickle',mode = 'rb') as f:
        mask_anchor = pickle.load(f)
    mask_anchor = np.reshape(mask_anchor,[361,5,1000,1000])
    mask__ = np.reshape(mask_anchor,[1805,1000,1000])
    mask_ = list()
    for i in range(1805):
        mask_parts = cv2.resize(mask__[i],(250,250))
        mask_parts[mask_parts>0] = 1
        mask_.append(mask_parts)
    mask_ = np.array(mask_)
    iou_list = list()
    iou_affine_list = list()
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
            #pdb.set_trace()
            or_ = np.sum(np.sum(np.sum(or_,2),2),1)
            and_ = np.sum(np.sum(np.sum(and_,2),2),1)
            iou = and_/or_
            #pdb.set_trace()
            max_index = np.argmax(iou)
            len_ann = len(annotations_x)
            len_anc = [len(np.where(mask_anchor[max_index][i]>0)[0]) for i in range(5)]
            idx = np.abs(np.array(len_anc)-len_ann).argmin()
            iou = np.sum(np.logical_and(np.reshape(mask_,[361,5,250,250])[max_index][idx],mask_annotation[0][0]))/np.sum(np.logical_or(np.reshape(mask_,[361,5,250,250])[max_index][idx],mask_annotation[0][0]))
            print(ann_0_len)
            print('iou  : {0}'.format(iou))
            iou_list.append(iou)
            R_list = list()
            T_list = list()
            #pdb.set_trace()
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
            if iou>0.75:
                R = 0
                T = [0.0,0.0]
            else:
                #pdb.set_trace()
                R, T = ICP_matching(ann_stack,anchor_stack)
            print(R)
            print(T)
            with open('../data/ann_anchor_data/mask_anchor_k.pickle',mode = 'rb') as f:
                anchor_ = pickle.load(f)
            anchor_ = np.reshape(anchor_,(1805,1000,1000))
            img = cv2.imread('../data/VOC2012/sphere_data/{0}'.format(img_name))
            with open('../data/mask_ann/{0}_{1}.pickle'.format(img_name[:6],ann_0_len),mode = 'rb') as f:
                an = pickle.load(f)
            #pdb.set_trace()
            X = np.zeros((1000,1000))
            X[ann[ann_len][1][ann_0_len][1]] = 255
            #pdb.set_trace()
            #X = cv2.resize(an,(1000,1000))*255
            #X[ann[0][1][0][1]] = 255
            #X[ann[0][1][0][1]] = 255
            #X[np.where(cv2.resize(an,(1000,1000))==1)] = 1
            an = cv2.resize(an,(1000,1000))*255
            #pdb.set_trace()

            an_ = mask_anchor[max_index][idx]
            affine = np.array([[1,0,T[0]],[0,1,-T[1]]])
            pre=cv2.warpAffine(an_, affine, (1000,1000))
            affine = cv2.getRotationMatrix2D((0,0),-R,1.0)
            pre = cv2.warpAffine(an_,affine,(1000,1000))
            pre_resize = cv2.resize(pre,(250,250))
            pre_resize[pre_resize>0] = 1

            or_ = np.sum(np.logical_or(pre_resize,mask_annotation[0]))
            and_ = np.sum(np.logical_and(pre_resize,mask_annotation[0]))
            iou_affine = and_/or_
            #pdb.set_trace()
            iou_affine_list.append(iou_affine)
            print('affine iou:{0}'.format(iou_affine))
            #pdb.set_trace()
            #pre = an_

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
            #pdb.set_trace()
            pre = np.transpose(pre,[1,2,0])
            X = np.tile(np.transpose(X[np.newaxis],[1,2,0]),[1,1,3])
            #pre = np.tile(pre[newaxis],[])
            #pdb.set_trace()
            prediction = cv2.addWeighted(np.asarray(img,np.float64),0.7,np.asarray(pre,np.float64),0.3,0)
            prediction = cv2.addWeighted(np.asarray(prediction,np.float64),0.6,np.asarray(X,np.float64),0.4,0)
            cv2.imwrite('../../GoogleDrive/messigray_n_{0}_{1}.png'.format(ann_len,ann_0_len),prediction)
            cv2.imwrite('messigray_{0}_{1}.png'.format(ann_len,ann_0_len),prediction)
            cv2.imwrite('sample_ann.png',X)
            """
            not_affine = np.append(an_[np.newaxis],np.zeros((1,1000,1000)),0)
            pre_2 = np.zeros((1000,1000))
            pre_2[np.where(an_>0)]= 200
            pre_2 = pre_2[np.newaxis]
            not_affine = np.transpose(np.append(not_affine,pre_2,0),[1,2,0])
            #pdb.set_trace()
            prediction = cv2.addWeighted(np.asarray(img,np.float64),0.7,np.asarray(not_affine,np.float64),0.3,0)
            prediction = cv2.addWeighted(np.asarray(prediction,np.float64),0.6,np.asarray(X,np.float64),0.4,0)
            cv2.imwrite('../../GoogleDrive/not_affine_{0}_{1}.png'.format(ann_len,ann_0_len),prediction)
            ###############################
            #pdb.set_trace()
            """

            current = [name,R,T,x_min,y_min,x_max,y_min,max_index*idx]
            #pdb.set_trace()
            all.append(current)
        #pdb.set_trace()
        add = [[img_name,[all]]]
        dumps += add
        #pdb.set_trace()
        if ann_len == 0:
            print('iou_mean:{0}'.format(np.mean(np.array(iou_list))))
            print('iou_affine_mean:{0}'.format(np.mean(np.array(iou_affine_list))))
            pdb.set_trace()
        print("finish:{0}_{1}".format(ann_len,path_num))
        if ann_len % 50 == 0:
            with open('../data/{0}/redidual_parts_{1}.pickle'.format(path[path_num],ann_len//50),mode = 'wb') as f:
                pickle.dump(dumps,f)
            dumps = list()

    with open('../data/{0}/redidual_1.pickle'.format(path[path_num]),mode = 'wb') as f:
            pickle.dump(dumps,f)

    return 0



def make_area():
    file = ['ann_coords_1','ann_coords_2','ann_coords_3','ann_coords_4']
    for i in file:
        with open('../data/ann_anchor_data/{0}.pickle'.format(i),mode = 'rb') as f:
            ann = pickle.load(f)


        for ann_len in range(len(ann)):
            for ann_0_len in range(len(ann[ann_len][1])):
                mask = np.zeros((1000,1000))
                X=ann[ann_len][1][ann_0_len][1]
                mask[X] = 1
                name = ann[ann_len][0][:6]
                #pdb.set_trace()
                mask = cv2.resize(mask,(70,70))
                with open('../data/mask_ann/{0}_{1}.pickle'.format(name,ann_0_len),mode = 'wb') as f:
                    pickle.dump(mask,f)
            print("finish : {0}".format(name))

if __name__ ==  '__main__':

    if not os.path.exists('../data/ann_anchor_data/mask_anchor.pickle'):#mask anchorが存在しない場合、mask anchorの作成
        with open("anchor_kmeans.txt") as f:
            x = f.read().split()

        anchors = mask_anchor(np.array(x),19)

        #anchors = mask_anchor(anchors,19)

        with open('../data/ann_anchor_data/mask_anchor_k.pickle',mode = 'wb') as f:
            pickle.dump(anchors,f)
    #----------------------------------------
    #mask anchorを座標系に変換:(1000,1000)→[x座標][y座標]
    if not os.path.exists('../data/ann_anchor_data/anchor_coords.pickle'):
        print("make mask anchor")

        with open('../data/ann_anchor_data/mask_anchor_k.pickle',mode = 'rb') as f:
            anchor = pickle.load(f)

        anchor_coords=make_coords_from_mask(anchor,0)#mask anchorを座標系に変換

        with open('../data/ann_anchor_data/anchor_coords_k.pickle',mode = 'wb') as f:
            pickle.dump(anchor_coords,f)
        print("finish making mask anchor")

    #----------------------------------------

    #----------------------------------------
    #マスクアノテーションが存在しなければ作成し、pickleファイルで保存
    #ファイルがあれば読み込み
    if not os.path.exists('../data/ann_anchor_data/ann_coords_1.pickle'):
        print("make mask annotations_1")
        path = '../data/VOC2012/AnnotationsTrain_1' #残差を計算したい対象
        pick = ['car','Truck'] #見つけたい物体

        annotations = pascal_voc_clean_xml(path,pick)#データの読み込み、mask annotationsの作成

        print("make mask annotations coords_1")

        ann_coords=make_coords_from_mask(annotations,1)#mask annotationsを座標系に変換

        with open('../data/ann_anchor_data/ann_coords_1_T.pickle',mode = 'wb') as f:
            pickle.dump(ann_coords,f)
        print("finish making mask annotations coords")

        del annotations,ann_coords

    #elif not os.path.exists('mask_annotations_2.pickle'):
        print("make mask annotations_2")
        path = '../data/VOC2012/AnnotationsTrain_2' #残差を計算したい対象
        pick = ['car','Truck'] #見つけたい物体
        annotations = pascal_voc_clean_xml(path,pick)

        ann_coords=make_coords_from_mask(annotations,1)

        with open('../data/ann_anchor_data/ann_coords_2_T.pickle',mode = 'wb') as f:
            pickle.dump(ann_coords,f)
        print("finish making mask annotations coords")

        del annotations,ann_coords

    #elif not os.path.exists('mask_annotations_3.pickle'):
        print("make mask annotations_3")
        path = '../data/VOC2012/AnnotationsTrain_3/AnnotationsTrain_1' #残差を計算したい対象
        pick = ['car','Truck'] #見つけたい物体
        annotations = pascal_voc_clean_xml(path,pick)

        ann_coords=make_coords_from_mask(annotations,1)

        with open('../data/ann_anchor_data/ann_coords_3_T.pickle',mode = 'wb') as f:
            pickle.dump(ann_coords,f)
        print("finish making mask annotations coords")

        del annotations,ann_coords

    #elif not os.path.exists('mask_annotations_4.pickle'):
        print("make mask annotations_4")
        path = '../data/VOC2012/AnnotationsTrain_4/AnnotationsTrain_2' #残差を計算したい対象
        pick = ['car','Truck'] #見つけたい物体
        annotations = pascal_voc_clean_xml(path,pick)

        ann_coords=make_coords_from_mask(annotations,1)

        with open('../data/ann_anchor_data/ann_coords_4_T.pickle',mode = 'wb') as f:
            pickle.dump(ann_coords,f)
        print("finish making mask annotations coords")

        del annotations,ann_coords


    #教師データの作成
    #anchor coordsとannotation coordsを用いてICPマッチング
    #実行に1日くらいかかるのでプログラム分割して実行可能。
    #detect_R_Tの説明は本プログラムを参考に動作を確認してください
    #pre_0.py,pre_1.py,pre_2.py,pre_3.py
    if not os.path.exists('../data/redidual_1/redidual_parts_1.pickle'):
        with open('../data/ann_anchor_data/anchor_coords_k.pickle',mode = 'rb') as f:
            anchor = pickle.load(f)

        with open('../data/ann_anchor_data/ann_coords_1.pickle',mode = 'rb') as f:
            ann_1 = pickle.load(f)

        print("start detect the redidual between anchors and annotations")
        ann_1 = detect_R_T(ann_1,anchor,0)#最もIoUが高いアンカーを選択し、ICPマッチングを行う

        print("finish 1")

        with open('../data/ann_anchor_data/ann_coords_2.pickle',mode = 'rb') as f:
            ann_1 = pickle.load(f)
        ann_1 = detect_R_T(ann_1,anchor,1)

        print("finish 2")

        with open('../data/ann_anchor_data/ann_coords_3.pickle',mode = 'rb') as f:
            ann_1 = pickle.load(f)
        ann_1 = detect_R_T(ann_1,anchor,2)

        print("finish 3")

        with open('../data/ann_anchor_data/ann_coords_4.pickle',mode = 'rb') as f:
            ann_1 = pickle.load(f)
        ann_1 = detect_R_T(ann_1,anchor,3)

        print("finish 4")

    if not os.path.exists('../data/ann_anchor_data/annotations_nor_.pickle'):
    #Tの正規化
    #正規化しなければ、Rの学習が進みません

        dumps = list()
        dumps_1,cur_dir = load_data('../data/redidual_1_an')
        os.chdir(cur_dir)
        dumps_2,cur_dir = load_data('../data/redidual_2_an')
        os.chdir(cur_dir)
        dumps_3,cur_dir = load_data('../data/redidual_3_an')
        os.chdir(cur_dir)
        dumps += dumps_1
        dumps += dumps_2
        dumps += dumps_3

        t_0_max = -100000
        t_0_min = 100000
        t_1_max = -100000
        t_1_min = 100000
        #pdb.set_trace()
        annotations = dumps

        T_0 = []
        T_1 = []
        R = []
        #pdb.set_trace()
        for i in range(len(dumps)):
            for j in range(len(dumps[i][1][0])):
                #pdb.set_trace()
                T_0.append(dumps[i][1][0][j][2][0])
                T_1.append(dumps[i][1][0][j][2][1])
                #pdb.set_trace()
                R.append(dumps[i][1][0][j][1])
            #print(i)
        #pdb.set_trace()
        #sns.set_style("whitegrid")
        T_0 = np.array(T_0)
        T_1 = np.array(T_1)
        R = np.array(R)
        #pdb.set_trace()
        """        
        #Tの正規化前の分布確認用
        plt.hist(T_0)
        #plt.plot(np.array(T_0))
        plt.savefig('../../GoogleDrive/T_0_not_nor_k.png')
        plt.clf()
        plt.hist(T_1)
        #plt.plot(np.array(T_1))
        plt.savefig('../../GoogleDrive/T_1_not_nor_k.png')
        plt.clf()
        plt.hist(R)
        plt.savefig('../../GoogleDrive/R_.png')
        plt.clf()
        """
        #pdb.set_trace()
        T_0_mean = np.mean(T_0)
        T_0_var = np.var(T_0)

        T_1_mean = np.mean(T_1)
        T_1_var = np.var(T_1)

        R_mean = np.mean(R)
        R_var = np.var(R)

        print('T_0   mean:{0}  var:{1}'.format(T_0_mean,T_0_var))
        print('T_1   mean:{0}  var:{1}'.format(T_1_mean,T_1_var))
        print('R     mean:{0}  var:{1}'.format(R_mean,R_var))

        t_0_max = np.max(np.array(T_0))
        t_0_min = np.min(np.array(T_0))
        t_1_max = np.max(np.array(T_1))
        t_1_min = np.min(np.array(T_1))
        for i in range(len(annotations)):
            for j in range(len(annotations[i][1][0])):
                #pdb.set_trace()
                X_0 = np.array(annotations[i][1][0][j][2][0])
                X_1 = np.array(annotations[i][1][0][j][2][1])
                #pdb.set_trace()
                X_0 = ((X_0-t_0_min)/(t_0_max-t_0_min))*2 - 1#最大値を1,最小値-1に設定
                X_1 = ((X_1-t_1_min)/(t_1_max-t_1_min))*2 - 1
                annotations[i][1][0][j][2] = np.array((X_0,X_1)).T.tolist()
        max_min = [t_0_max,t_0_min,t_1_max,t_1_min]

        with open('../data/ann_anchor_data/annotations_only_iou.pickle',mode = 'wb') as f:
            pickle.dump(annotations,f)
        with open('../data/ann_anchor_data/max_min_k.pickle',mode = 'wb') as f:
            pickle.dump(max_min,f)
        #------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------
        #cfgにmin,maxの値が書き込まれます。以前のものと混同してしまうので、trainに入る前に以前のものは消去する
        #------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------
        with open('../cfg/tiny-without-iou.cfg','a') as f:
            print("t_0_max = {0}".format(t_0_max),file = f)
            print("t_0_min = {0}".format(t_0_min),file = f)
            print("t_1_max = {0}".format(t_1_max),file = f)
            print("t_1_min = {0}".format(t_1_min),file = f)

        """#test_dataの分割
        annotations = glob.glob('../data/redidual_4/*.pickle')
        for i,file in enumerate(annotations):

            with open(file,mode = 'rb') as f:
                annotations_parts = pickle.load(f)
            for j in range(len(annotations_parts)):
            #pdb.set_trace()
                name = annotations_parts[j][0]
                new = '../data/VOC2012/sphere_data/'+ name
                try:
                    new_path = shutil.move(new, '../data/VOC2012/sphere_test/')
                except:
                    pass
                print(name)

        """
