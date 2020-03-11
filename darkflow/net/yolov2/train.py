import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from ..yolo.misc import show
import numpy as np
import os
import math
import pdb
import cv2
from tensorflow.python.ops import tensor_array_ops
from tensorflow .python import debug as tf_debug
import pickle


def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """

    # meta
    m = self.meta

    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W # number of grid cells
    anchors = m['anchors']

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]
    size4 = [None, HW, B, 1]

    # return the below placeholders
    _probs    =    tf.placeholder(tf.float32, size1)
    _confs    =    tf.placeholder(tf.float32, size2)
    # weights term for L2 loss
    _proid    =    tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _R        =    tf.placeholder(tf.float32, size4 )
    _T        =    tf.placeholder(tf.float32, size2 + [2])

    self.placeholders = {
        'probs':_probs, 'confs':_confs, 'R':_R, 'T':_T, 'proid':_proid
    }

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (1 + 2 + 1 + C)])
    R = tf.reshape(net_out_reshape[:, :, :, :, 0],[-1,H*W,B,1]) #出力から回転角度Rを抽出
    T = tf.reshape(net_out_reshape[:, :, :, :, 1:3],[-1,H*W,B,2]) #出力から回転角度Tを抽出

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 3]) #出力からconfidenceを抽出 expit_tensorによって0~1で表現
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 4:]) #出力から各カテゴリに属する確率
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])
    adjusted_net_out = tf.concat([T,adjusted_c, adjusted_prob], 3) #T,confidence,クラスをconcat
    batch_size = tf.shape(adjusted_net_out)[0] #batch_sizeの取得

    confs = _confs
    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs #重みの計算
    weight_coo = tf.concat(2 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    self.fetch += [_probs, confs, conid, cooid, proid]

    R = tf.reshape(R,[batch_size,H*W,B,1])
    _R = tf.reshape(_R,[batch_size,H*W,B,1]) #Rの真値

    true = tf.concat([_T,tf.expand_dims(confs, 3), _probs ], 3) #T,confidence,クラス確率の真値
    wght = tf.concat([cooid,tf.expand_dims(conid, 3), proid ], 3) #T,confidence,クラス確率の予測データ
    pre_angle = tf.concat([tf.math.sin(R),tf.math.cos(R)],3) #Rの予測データ
    true_angle = tf.concat([tf.math.sin(_R),tf.math.cos(_R)],3)#Rの教師データ

    pre_norm = tf.reshape(tf.norm(pre_angle,axis=3),(batch_size,H*W,B,1))
    true_norm = tf.reshape(tf.norm(true_angle,axis=3),(batch_size,H*W,B,1))

    vec_dot = tf.matmul(pre_angle,true_angle,transpose_b=True)
    vec_dot = vec_dot * np.reshape(np.eye(B,B), [1, 1, B, B])
    vec_dot = tf.reshape(tf.reduce_sum(vec_dot,axis=3),(batch_size,H*W,B,1))
    vec_abs_fin = tf.add(tf.multiply(pre_norm,true_norm),0.00001)#biternion loss

    difal = tf.subtract(1., tf.divide(vec_dot, vec_abs_fin))
    weight_difal = tf.concat(1 * [tf.expand_dims(confs, -1)], 3)
    difal = difal * weight_difal#biternion lossに重みをかける

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true,2)

    loss = tf.multiply(loss, wght)

    loss = tf.concat([loss,difal],3)
    #pdb.set_trace()
    loss = tf.reshape(loss, [-1, H*W*B*(3 + 1 + C)])

    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)

def expit_tensor(x):
    return 1. / (1. + tf.exp(-x))

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 5, 5], strides=[1, 1, 5, 5], padding='SAME')
