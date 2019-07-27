import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from ..yolo.misc import show
import numpy as np
import os
import math
import pdb
def expit_tensor(x):
	return 1. / (1. + tf.exp(-x))

def mask_anchor(anchor,H):
    img_x = 1000
    img_y = 1000

    step_x=0
    step_y=0
    S = H
    anchor = np.reshape(anchor,(5,2))
    anchor_size = anchor.shape[0]
    step_size = int(img_x/S)

    #pdb.set_trace()

    mask = np.array([])

    for i in range(S):
        for t in range(S):
            if t==0:
                step_x = 0
            center_x = int((step_x + (step_x + step_size))/2)
            center_y = int((step_y + (step_y + step_size))/2)
            for l in range(anchor_size):
                w_ = anchor[l][0]
                h_ = anchor[l][0]

                mask_base = np.zeros((img_x,img_y),dtype=int)

                side = int(w_*500/39)
                ver = int(h_*200/13)

                #pdb.set_trace()

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


                mask_base[ver_min:ver_max,side_min:side_max] = 255
                resize_mask = np.resize(mask_base,(19,19))
                if l == 0:
                    mask = resize_mask[np.newaxis]
                else:
                    mask = np.append(mask,resize_mask[np.newaxis],axis=0)
                print(l)
                #pdb.set_trace()

            step_x += step_size
            #pdb.set_trace()
            if t == 0:
                mask_ = mask[np.newaxis]
            else:
                mask_ = np.append(mask_,mask[np.newaxis],axis=0)

        step_y += step_size
        if i == 0:
            mask_fi = mask_[np.newaxis]
        else:
        	mask_fi = np.append(mask_fi,mask_[np.newaxis],axis=0)

    return mask_fi

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

    anchor = mask_anchor(anchors,H)
    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]
    #pdb.set_trace()
    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [1])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    self.placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright
    }
    #pdb.set_trace()

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (HW + 1 + C)]) #mask+信頼度+クラス

    #pdb.set_trace()
    coords = net_out_reshape[:, :, :, :, :1]
    coords = tf.reshape(coords, [-1, H*W, B, 1]) #<tf.Tensor 'Reshape_1:0' shape=(?, 361, 5, 1) dtype=float32>
    #adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])
    #adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    #coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)
    pdb.set_trace()
    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 1])
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1]) #<tf.Tensor 'Reshape_2:0' shape=(?, 361, 5, 1) dtype=float32>
    #pdb.set_trace()
	#####################################################################
	#####################################################################
	###########ここからlossの設計とIOUの設定###############################
	#####################################################################
	#####################################################################

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 2:])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C]) #<tf.Tensor 'Reshape_3:0' shape=(?, 361, 5, 2) dtype=float32>

    adjusted_net_out = tf.concat([coords, adjusted_c, adjusted_prob], 3) #<tf.Tensor 'concat_2:0' shape=(?, 361, 5, 4) dtype=float32>

    area_pred_mask = coords
    #pdb.set_trace()
    _areas = tf.reshape(_areas,[-1,H*W,B,1])
    sum_pred_true = tf.add(_areas,area_pred_mask)
    threshold_tensor = tf.ones_like(area_pred_mask)*300
    zeros_tensor = tf.zeros_like(area_pred_mask)
    intersect = tf.subtract(sum_pred_true,threshold_tensor)
    intersect = tf.maximum(intersect,zeros_tensor)
    intersect = tf.math.divide(tf.subtract(intersect,tf.reduce_min(intersect)),tf.subtract(tf.reduce_max(intersect),tf.reduce_min(intersect)))*255
    intersect = intersect[:,:,:,0]
    area_pred_mask = area_pred_mask[:,:,:,0]
    _areas = _areas[:,:,:,0]
    #pdb.set_trace()
	#sample = (sample - sample_min) / (sample_max - sample_min)
	#min = tf.reduce_min



    #wh = tf.pow(coords[:,:,:,2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
    #area_pred = wh[:,:,:,0] * wh[:,:,:,1]
    #centers = coords[:,:,:,0:2]
    #floor = centers - (wh * .5)
    #ceil  = centers + (wh * .5)

    # calculate the intersection areas
	# 交差面積の計算
    """
    intersect_upleft   = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])
    """

    # calculate the best IOU, set 0.0 confidence for worse boxes
	# bestのIOUを計算し、一番悪いボックスには信頼度0.0を設定
    """
    iou = tf.truediv(intersect, _areas + area_pred - intersect)　#tf.truediv:割算 intersect/(_areas + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True)) #tf.reduce_max:Tensorの要素の最大を計算
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)
    """

    iou =  tf.truediv(intersect, _areas + area_pred_mask - intersect)
    best_box = tf.equal(iou,tf.reduce_max(iou,[2],True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box,_confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)

    #pdb.set_trace()

    print('Building {} loss'.format(m['model']))
    """
    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    """
    smooth = 1
    y_pred = tf.reshape(area_pred_mask,[-1])/255
    y_true = tf.reshape(_areas,[-1])/255
    intersection = tf.reduce_sum(y_pred*y_true)
    loss = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + loss
    pdb.set_trace()
    self.loss = .5 * tf.reduce_mean(loss)

    tf.summary.scalar('{} loss'.format(m['model']), self.loss)
