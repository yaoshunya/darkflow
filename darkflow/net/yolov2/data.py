from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from ..yolo.predict import preprocess
from ..yolo.data import shuffle
from copy import deepcopy
import pickle
import numpy as np
import os
import pdb
from PIL import Image
import math

def _batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's
    input & loss layer correspond to this chunk
    """
    meta = self.meta
    labels = meta['labels']

    H, W, _ = meta['out_size']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']

    # preprocess

    jpg = chunk[0]
    w = 1000
    h = 1000
    allobj_ = chunk[1][0]
    allobj = deepcopy(allobj_)
    path = os.path.join(self.FLAGS.dataset, jpg)
    img = self.preprocess(path, allobj)

    # Calculate regression target
    cellx = 1. * w / W #画像の横幅を１グリッドあたりのピクセル数
    celly = 1. * h / H #画像の縦幅１グリッドあたりのピクセル数



    # Calculate placeholders' values
    probs = np.zeros([H*W,B,C])  #361x5x2セルごとの各クラスへの所属確率
    confs = np.zeros([H*W,B]) #361x5 セルごとの各BBの信頼度
    R = np.zeros([H*W,B,1]) #回転角度
    T = np.zeros([H*W,B,2])#並進ベクトルT
    proid = np.zeros([H*W,B,C])


    for obj in allobj:
            
        #areas[mod,q, :] = area*255#真値のマスク：confidenceの与え方の変更により、使用しない。
        probs[obj[3],obj[4], :] = [[0.]*C][0]  #物体があるセルにクラスの数だけ要素を設けている
        probs[obj[3],obj[4], labels.index(obj[0])] = 1.  #そのうち入力された物体の方の確率を１とする
        proid[obj[3],obj[4], :] = [[1.]*C][0]
        R[obj[3],obj[4], :] = np.array(math.radians(obj[1]))
        T[obj[3],obj[4], :] = np.array(obj[2])
        confs[obj[3],obj[4]] = 1.  #物体が存在するセルの各BBの信頼度を１とする
        #obj[7]には、最も真値に近いアンカーのindexが入っている。
    #areas = np.reshape(areas,[H*W,B,70,70])
    probs = np.reshape(probs,[H*W,B,C])
    proid = np.reshape(proid,[H*W,B,C])
    R = np.reshape(R,[H*W,B,1])
    T = np.reshape(T,[H*W,B,2])
    confs = np.reshape(confs,[H*W,B])

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'R': R, 'T': T, 'proid': proid,
    }

    return inp_feed_val, loss_feed_val
