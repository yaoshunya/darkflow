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
import final_prepare as fp
#  ICP parameters
EPS = 0.00001
MAXITER = 100
dataPath='../data'
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
    icpマッチングを行うことでマスクアンカー、アノテーションとの残差を計算
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

        if dError <= EPS:
            break
        elif MAXITER <= count:
            print("Not Converge...", error, dError, count)
            break

    R = np.array(H[0:2, 0:2])
    T = np.array(H[0:2, 2])

    try:
        R = math.degrees(math.acos(R[0][0]))
    except:
        R = 0

    return R,T


def update_homogeneous_matrix(Hin, R, T):
    #icpマッチングでのアフィン行列の更新
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
    #ppointsとcpointsの点どうしを対応付ける
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
    #特異値分解を用いたicpマッチングを行う
    pm = np.mean(ppoints, axis=1)
    cm = np.mean(cpoints, axis=1)

    pshift = ppoints - pm[:, np.newaxis]
    cshift = cpoints - cm[:, np.newaxis]

    W = cshift @ pshift.T
    u, s, vh = np.linalg.svd(W)

    R = (u @ vh).T
    t = pm - (R @ cm)

    return R, t

def mask_anchor(anchor,H):
    img_x = 1000#画像の幅
    img_y = 1000#画像の高さ

    step_x=0
    step_y=0
    S = H
    #anchor = np.reshape(anchor,(5,2))
    anchor_size = anchor.shape[0]
    step_size_x = int(img_x/S)#グリッドに分割した際のグリッド1つ当たりのピクセル数
    step_size_y = int(img_y/S)

    mask = np.array([])

    for i in range(S):
        for t in range(S):
            if t==0:
                step_x = 0
            center_x = int((step_x + (step_x + step_size_x))/2)#gridの中心座標x
            center_y = int((step_y + (step_y + step_size_y))/2)#gridの中心座標y
            for l in range(anchor_size):
                #pdb.set_trace()
                w_ = float(x[l].split(",")[0])
                h_ = float(x[l].split(",")[1])

                mask_base = np.zeros((img_x,img_y),dtype=int)

                side = int(w_)/2
                ver = int(h_)/2

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


                mask_base[int(ver_min):int(ver_max),int(side_min):int(side_max)] = 255#mask anchor(歪みなし)の作成
                #mask_base = np.resize(mask_base,(500,500))
                #-------------------------------------------------
                #アンカーの球面写像
                grid = fp.get_projection_grid(b=500)
                rot = fp.rand_rotation_matrix(deflection=1.0)
                grid = fp.rotate_grid(rot,grid)
                mask_base = fp.project_2d_on_sphere(mask_base,grid)#mask anchorに歪みを持たせる
                #-------------------------------------------------

                resize_mask = mask_base.T
                if l == 0:
                    mask = resize_mask[np.newaxis]
                else:
                    mask = np.append(mask,resize_mask[np.newaxis],axis=0)
            step_x += step_size_x
            if t == 0:
                mask_ = mask[np.newaxis]
            else:
                mask_ = np.append(mask_,mask[np.newaxis],axis=0)
        print(i)
        step_y += step_size_y
        if i == 0:
            mask_fi = mask_[np.newaxis]
            #mask_fi_row = mask__row[np.newaxis]
        else:
            mask_fi = np.append(mask_fi,mask_[np.newaxis],axis=0)
    return mask_fi

def pascal_voc_clean_xml(ANN, pick, exclusive = False):
    #mask annotationの球面写像
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')
    size = len(annotations)
    mask = np.array([])
    H = 19
    W = 19
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

            centerx = .5*(xx+xn)
            centery = .5*(yn+yx)

            cellx = 1242/W
            celly = 375/H

            cx = centerx/cellx
            cy = centery/celly

            if cx>=W or cy>=H:index = None

            index = int(np.floor(cy)*W + np.floor(cx))#グリッド上のどこに存在するのかを求める
            #make mask annotation from coordinates
            ###############################################################
            mask_prepare = np.zeros((375,1242),dtype=int)
            mask_prepare[yn:yx,xn:xx]=255
            #pdb.set_trace()
            mask_parts = np.array([])
            grid = fp.get_projection_grid(b=500)
            rot = fp.rand_rotation_matrix(deflection=1.0)
            grid = fp.rotate_grid(rot,grid)
            mask_parts = fp.project_2d_on_sphere(mask_prepare,grid)
            #mask_parts = cv2.resize(mask_parts,(19,19))
            mask_parts = np.reshape(mask_parts,(1000,1000)).T

            mask_ = mask_parts[np.newaxis]
            current = [name,mask_,index]
            all += [current]

        add = [[jpg,all]]

        dumps += add
        in_file.close()
        ###################################################################

    os.chdir(cur_dir)
    all = list()
    current = list()
    add = list()
    return dumps

def make_up_left_coord(H):
    #球面グリッドの左上の座標を取得
    step = 1000//H
    coord_list = list()
    print('make spherical up left coord')
    for i in range(H):
        for j in range(H):
            base = np.zeros((1000,1000))
            base[step*i:step*i+4,step*j:step*j+4] = 255#グリッドの該当箇所を白塗り
            grid = fp.get_projection_grid(b=500)
            rot = fp.rand_rotation_matrix(deflection=1.0)#球面写像
            grid = fp.rotate_grid(rot,grid)
            rotate_base = fp.project_2d_on_sphere(base,grid).T
            x_min = np.min(np.where(rotate_base>0)[0])#白塗りの左上の座標を取得
            y_min = np.min(np.where(rotate_base>0)[1])
            coord = [y_min,x_min]
            coord_list.append(coord)
            #pdb.set_trace()
            print('finish:{0}'.format(i*19+j))
    with open('{0}/ann_anchor_data/upleft_coord.pickle'.format(dataPath),mode = 'wb') as f:
        pickle.dump(coord_list,f)


def make_coords_from_mask(data,flag):

    if flag == 0:#anchorを座標系に変換する場合
        anchor = data
        anchor = np.reshape(anchor,[anchor.shape[0]*anchor.shape[1],anchor.shape[2],anchor.shape[3],anchor.shape[4]])
        anchor_coords = list()

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

        for i in range(len(annotations)):
            jpg = annotations[i][0]
            all = []

            for j in range(len(annotations[i][1])):
                name = annotations[i][1][j][0]
                current = list()
                ann_coords_parts = np.where(np.reshape(annotations[i][1][j][1],[1000,1000])>0)
                ann_coords_parts = (ann_coords_parts[0],ann_coords_parts[1])
                current = [name,ann_coords_parts,annotations[i][1][j][2]]
                all += [current]

            add = [[jpg,all]]
            ann_coords += add

        return ann_coords
    return 0

def detect_R_T(ann,anchor,path_num):
    #教師データの作成

    B = 10
    H = 19
    W = 19

    img_x = 1000
    img_y = 1000
    img_x_resize = 250
    img_y_resize = 250
    dumps = list()
    path = ['redidual_1','redidual_2','redidual_3','redidual_4']
    #mask anchorの読み込み
    with open('{0}/ann_anchor_data/mask_anchor_k.pickle'.format(dataPath),mode = 'rb') as f:
        mask_anchor = pickle.load(f)
    mask_anchor = np.reshape(mask_anchor,[H*W,B,img_x,img_y])
    mask__ = np.reshape(mask_anchor,[H*W*B,img_x,img_y])
    mask_ = list()

    #iouを計算する際に計算量を削減する為に(250,250)に変換
    for i in range(H*W*B):
        mask_parts = cv2.resize(mask__[i],(img_x_resize,img_y_resize))
        mask_parts[mask_parts>0] = 1
        mask_.append(mask_parts)

    mask_ = np.reshape(np.array(mask_),[H*W,B,img_x_resize,img_y_resize])

    #グリッドを球面に写像した際の左上の座標
    with open('{0}/ann_anchor_data/upleft_coord.pickle'.format(dataPath),mode='rb') as f:
        upleft_coord = pickle.load(f)

    iou_list = list()
    iou_affine_list = list()
    kernel = [[1,1,1],[1,-8,1],[1,1,1]] #エッジ検出する際に用いるカーネルサイズ
    T_0_list = list()
    T_1_list = list()
    for ann_len in range(len(ann)):
        per = (ann_len/len(ann))*100
        print('progress per :{0} %'.format(per))
        img_name = ann[ann_len][0]
        all = list()
        print(img_name)
        for ann_0_len in range(len(ann[ann_len][1])):

            error = list()
            iou = list()
            name = ann[ann_len][1][ann_0_len][0]
            current = list()

            X = np.zeros((img_x,img_y))
            X[ann[ann_len][1][ann_0_len][1]] = 1 #annotation座標からmask annotationの作成
            X = cv2.resize(X,(img_x_resize,img_y_resize))
            X[np.where(X>0)] = 1
            X = np.tile(X[np.newaxis],(B,1,1))

            annotations_x = np.array(ann[ann_len][1][ann_0_len][1][0])
            annotations_y = np.array(ann[ann_len][1][ann_0_len][1][1])
            index = ann[ann_len][1][ann_0_len][2] #pascal_voc_clean_xmlでさ作成した物体が存在するindex
            if index == None:
                #本家YOLOでも実装されていたのでこif分を入れたが、ここには入らない
                R = None
                T = None
                idx = None
                break
            upleft_now = upleft_coord[index]#球面グリッドの左上座標
            mask_anchor_now = mask_[index]#選択されたグリッド上にあるmask anchor(5,250,250)
            or_ = np.sum(np.sum(np.logical_or(X,mask_anchor_now),2),1)
            and_ = np.sum(np.sum(np.logical_and(X,mask_anchor_now),2),1)
            iou = and_/or_#annotationとanchorのiouを計算

            idx = np.argmax(iou)#最もIoUが高いアンカーを選択

            or_ = np.sum(np.logical_or(X,mask_[index][idx]))
            and_ = np.sum(np.logical_and(X,mask_[index][idx]))
            iou = and_/or_
            R_list = list()
            T_list = list()

            ann_now = ann[ann_len][1][ann_0_len][1]
            A = np.zeros((img_x,img_y),dtype = 'uint8')
            A[ann_now] = 255
            ann_now = A
            anchor_now = mask_anchor[index][idx]

            dst_ann = np.where(cv2.Laplacian(ann_now,cv2.CV_32F,ksize=3)>0) #エッジ検出
            dst_anchor = np.where(cv2.Laplacian(anchor_now,cv2.CV_32F,ksize=3)>0)

            dst_ann = [dst_ann[0]-upleft_now[0],dst_ann[1]-upleft_now[1]]#エッジ座標から球面グリッドの左上の座標を引く
            dst_anchor = [dst_anchor[0]-upleft_now[0],dst_anchor[1]-upleft_now[1]]

            ann_len_ = len(dst_ann[0])
            anchor_len_ = len(dst_anchor[0])

            if ann_len_ < anchor_len_:
                sample_len = ann_len_//10
            else:
                sample_len = anchor_len_//10
            if sample_len > 50:
                sample_len = 50
            print(sample_len)
            iou_affine = 0
            count = 0
            best_iou = iou
            best_R = 0.0
            best_T = [0.0,0.0]
            img = cv2.imread('{0}/kitti/sphere_data/{1}'.format(dataPath,img_name))
            
            mask_an = cv2.resize(ann_now,(img_x_resize,img_y_resize))
            mask_an[np.where(mask_an>0)] = 1
            while(iou_affine < 0.6):
            #条件を満たすまでicpマッチングを行い、最適なR,Tを求める
                my_list_ann = []
                my_list_anchor = []
                if iou_affine > 0.8:#iouが0.8以上であればok
                    break
                count += 1
                if count==50:
                    #50回試行してもうまくいかない場合は、最善のものを選択
                    if iou>best_iou:
                        R = 0.0
                        T = [0.0,0.0]
                        iou_affine = iou
                        break
                    else:
                        R = best_R
                        T = best_T
                        iou_affine = best_iou
                        break
                x = np.random.permutation(np.arange(ann_len_-1))[:sample_len]
                y = np.random.permutation(np.arange(anchor_len_-1))[:sample_len]
                
                ann_stack = np.vstack((dst_ann[0][x],dst_ann[1][x]))
                anchor_stack = np.vstack((dst_anchor[0][y],dst_anchor[1][y]))

                R,T  = ICP_matching(ann_stack,anchor_stack)

                X = np.zeros((img_x,img_y))
                X[ann[ann_len][1][ann_0_len][1]] = 255
                affine = cv2.getRotationMatrix2D((upleft_now[0],upleft_now[1]),R,1.0)#球面グリッドの左上の座標を中心として回転
                affine[0][2] += T[1]
                affine[1][2] += T[0]

                pre = cv2.warpAffine(anchor_now,affine,(img_x,img_y))
                pre_resize = cv2.resize(pre,(img_x_resize,img_y_resize))
                pre_resize[pre_resize>0] = 1

                or_ = np.sum(np.logical_or(pre_resize,mask_an))
                and_ = np.sum(np.logical_and(pre_resize,mask_an))
                iou_affine = and_/or_
                if best_iou<iou_affine:
                    best_iou = iou_affine
                    best_R = R
                    best_T = T
            print('iou       :{0}'.format(iou))
            print('affine iou:{0}'.format(iou_affine))
            print('T:        {0}'.format(T))
            T_0_list.append(T[0])
            T_1_list.append(T[1])
            if False:
                #教師データの作成ができているか確認用
                where_ = np.where(pre)
                pre_1 = np.zeros((img_x,img_y))
                pre_2 = np.zeros((img_x,img_y))
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

                #アフィン変換する前のアンカーを合成した画像を出力する用
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
                #pdb.set_trace()
                """
            current = [name,R,T,index,idx]
            all.append(current)

        add = [[img_name,[all]]]
        dumps += add
        if ann_len % 50 == 0:
            with open('{0}/{1}/redidual_parts_{2}.pickle'.format(dataPath,path[path_num],ann_len//50),mode = 'wb') as f:
                pickle.dump(dumps,f)
            dumps = list()

    with open('{0}/{1}/redidual_1.pickle'.format(dataPath,path[path_num]),mode = 'wb') as f:
            pickle.dump(dumps,f)

    return 0

if __name__ ==  '__main__':
    #python preprocess_ad.py
    #mask anchorの作成
    #mask annotationsの作成

    #python preprocess_ad.py 0
    #引数として0,1,2,3を指定
    #教師データが作成される

    #教師データ作成後
    #python preprocess_ad.py final
    #Tの正規化を行う
    path_coords = ['ann_coords_1_T','ann_coords_2_T','ann_coords_3_T','ann_coords_4_T']
    num_list = ['0','1','2','3']
    if len(sys.argv) == 1:#mask anchorが存在しない場合、mask anchorの作成
        """
        with open("anchor_kmeans.txt") as f:
            x = f.read().split()

        anchors = mask_anchor(np.array(x),19)

        #anchors = mask_anchor(anchors,19)

        with open('{0}/ann_anchor_data/mask_anchor_k.pickle'.format(dataPath),mode = 'wb') as f:
            pickle.dump(anchors,f)
        #----------------------------------------
        #mask anchorを座標系に変換:(1000,1000)→[x座標][y座標]

        print("make mask anchor")

        with open('{0}/ann_anchor_data/mask_anchor_k.pickle'.format(dataPath),mode = 'rb') as f:
            anchor = pickle.load(f)

        anchor_coords = make_coords_from_mask(anchor,0)#mask anchorを座標系に変換

        with open('{0}/ann_anchor_data/anchor_coords_k.pickle'.format(dataPath),mode = 'wb') as f:
            pickle.dump(anchor_coords,f)
        print("finish making mask anchor")

        #----------------------------------------

        path = ['AnnotationsTrain_1','AnnotationsTrain_2','AnnotationsTrain_3','AnnotationsTrain_4']
        pick = ['car','Truck'] #見つけたい物体
        #----------------------------------------
        #マスクアノテーションを作成し、pickleファイルで保存
        for i in range(len(path_coords)):
            print("make mask annotations_1")
            path_ = dataPath +'/kitti/' + path[i] #残差を計算したい対象
            annotations = pascal_voc_clean_xml(path_,pick)#データの読み込み、mask annotationsの作成
            print("make mask annotations {0}".format(i))

            ann_coords=make_coords_from_mask(annotations,1)#mask annotationsを座標系に変換
            path_ = dataPath + '/ann_anchor_data/' + path_coords[i] + '.pickle'
            with open(path_,mode = 'wb') as f:
                pickle.dump(ann_coords,f)
            print("finish making mask annotations coords")

            del annotations,ann_coords
        """
        make_up_left_coord(19)


    elif sys.argv[1] in num_list:

        with open('{0}/ann_anchor_data/anchor_coords_k.pickle'.format(dataPath),mode = 'rb') as f:
            anchor = pickle.load(f)
        #教師データの作成
        #anchor coordsとannotation coordsを用いてICPマッチング
        #実行に1日くらいかかるのでプログラム分割して実行可能。
        index = int(sys.argv[1])
        #pdb.set_trace()
        file_path = dataPath +'/ann_anchor_data/'+ path_coords[index] + '.pickle'
        with open(file_path,mode = 'rb') as f:
            ann_1 = pickle.load(f)
        print("start detect the redidual between anchors and annotations")
        ann_1 = detect_R_T(ann_1,anchor,index)#最もIoUが高いアンカーを選択し、ICPマッチングを行う

        print("finish {0}".format(index))


    elif sys.argv[1] == 'final':
    #Tの正規化
    #正規化しなければ、Rの学習が進みません
        dumps = list()
        dumps_1,cur_dir = load_data('{0}/redidual_1'.format(dataPath))
        os.chdir(cur_dir)
        dumps_2,cur_dir = load_data('{0}/redidual_2'.format(dataPath))
        os.chdir(cur_dir)
        dumps_3,cur_dir = load_data('{0}/redidual_3'.format(dataPath))
        os.chdir(cur_dir)
        dumps += dumps_1
        dumps += dumps_2
        dumps += dumps_3
        t_0_max = -100000
        t_0_min = 100000
        t_1_max = -100000
        t_1_min = 100000
        annotations = dumps

        T_0 = []
        T_1 = []
        R = []
        #pdb.set_trace()
        for i in range(len(dumps)):
            for j in range(len(dumps[i][1][0])):
                T_0.append(dumps[i][1][0][j][2][0])
                T_1.append(dumps[i][1][0][j][2][1])
                R.append(dumps[i][1][0][j][1])

        T_0 = np.array(T_0)
        T_1 = np.array(T_1)
        R = np.array(R)

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
        max_index = list()
        idx = list()
        for i in range(len(annotations)):
            for j in range(len(annotations[i][1][0])):
                X_0 = np.array(annotations[i][1][0][j][2][0])
                X_1 = np.array(annotations[i][1][0][j][2][1])
                X_0 = ((X_0-t_0_min)/(t_0_max-t_0_min))*2 - 1#最大値を1,最小値-1に設定
                X_1 = ((X_1-t_1_min)/(t_1_max-t_1_min))*2 - 1
                R = math.radians(annotations[i][1][0][j][1])
                annotations[i][1][0][j][2] = np.array((X_0,X_1)).T.tolist()
                annotations[i][1][0][j][1] = np.array(R)
                #pdb.set_trace()
                max_index.append(annotations[i][1][0][j][3])
                idx.append(annotations[i][1][0][j][4])
        max_min = [t_0_max,t_0_min,t_1_max,t_1_min]

        plt.hist(max_index)
        plt.savefig('../../GoogleDrive/max_index.png')
        plt.clf()

        plt.hist(idx)
        plt.savefig('../../GoogleDrive/idx.png')
        plt.clf()

        #pdb.set_trace()
        with open('{0}/ann_anchor_data/annotations_only_iou.pickle'.format(dataPath),mode = 'wb') as f:
            pickle.dump(annotations,f)
        with open('{0}/ann_anchor_data/max_min_k.pickle'.format(dataPath),mode = 'wb') as f:
            pickle.dump(max_min,f)
        """
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
        """

        """#test_dataの分割
        annotations = glob.glob('../data/redidual_4/*.pickle')
        for i,file in enumerate(annotations):

            with open(file,mode = 'rb') as f:
                annotations_parts = pickle.load(f)
            for j in range(len(annotations_parts)):
            #pdb.set_trace()
                name = annotations_parts[j][0]
                new = '../data/kitti/sphere_data/'+ name
                try:
                    new_path = shutil.move(new, '../data/kitti/sphere_test/')
                except:
                    pass
                print(name)

        """
