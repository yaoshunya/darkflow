from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from ..yolo.predict import preprocess
from ..yolo.data import shuffle
from copy import deepcopy
import pickle
import numpy as np
import os

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
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(self.FLAGS.dataset, jpg)
    img = self.preprocess(path, allobj)

    # Calculate regression target
    cellx = 1. * w / W #画像の横幅を１グリッドあたりのピクセル数
    celly = 1. * h / H #画像の縦幅１グリッドあたりのピクセル数
    for obj in allobj:
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax 物体の中心座標
        centery = .5*(obj[2]+obj[4]) #ymin, ymax 物体の中心座標
        cx = centerx / cellx #どこのセルにあるかの番号
        cy = centery / celly #どこのセルにあるかの番号
        if cx >= W or cy >= H: return None, None #１３以上なら画面外になってしまうから
        obj[3] = float(obj[3]-obj[1]) / w 
        obj[4] = float(obj[4]-obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery
        obj += [int(np.floor(cy) * W + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([H*W,B,C])  #169x5x2セルごとの各クラスへの所属確率
    confs = np.zeros([H*W,B]) #169x5 セルごとの各BBの信頼度
    coord = np.zeros([H*W,B,4])  #169x5x4  セルごとのBBの座標
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,4])
    for obj in allobj:
        probs[obj[5], :, :] = [[0.]*C] * B #物体があるセルにクラスの数だけ要素を設けている
        probs[obj[5], :, labels.index(obj[0])] = 1.  #そのうち入力された物体の方の確率を１とする
        proid[obj[5], :, :] = [[1.]*C] * B
        coord[obj[5], :, :] = [obj[1:5]] * B #中心ずれと幅高さ比率を、アンカーの数だけそれぞれに同じものを代入
        prear[obj[5],0] = obj[1] - obj[3]**2 * .5 * W # xleft BBの中心座標とBBの比率でそれぞれの座標を逆算
        prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * H # yup BBの中心座標とBBの比率でそれぞれの座標を逆算
        prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * W # xright BBの中心座標とBBの比率でそれぞれの座標を逆算
        prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * H # ybot BBの中心座標とBBの比率でそれぞれの座標を逆算
        confs[obj[5], :] = [1.] * B #物体が存在するセルの各BBの信頼度を１とする

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1) #単純にBBの左上の座標
    botright = np.expand_dims(prear[:,2:4], 1) #単純にBBの左上の座標
    wh = botright - upleft; #BBの縦横の幅
    area = wh[:,:,0] * wh[:,:,1] #セルに物体があった場合のBBの面積
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer 
    loss_feed_val = {
        'probs': probs, 'confs': confs, 
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft, 
        'botright': botright
    }

    return inp_feed_val, loss_feed_val

