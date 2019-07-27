import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from ..yolo.misc import show
import numpy as np
import os
import math
import pdb
import cv2

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


def expit_tensor(x):
	return 1. / (1. + tf.exp(-x))
"""
def shift_x_y(image, shift_x,shift_y):
    pdb.set_trace()
    h, w = image.shape[:2]
    anchor_size = image.shape[2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += shift_x #横シフトするピクセル値
    dest[:,1] += shift_y #縦シフトするピクセル値
    affine = cv2.getAffineTransform(src, dest)

    for i in range(h):
        for j in range(w):
            for k in range(anchor_size):
                im_anchor_parts = image[i][j][k]
                im_anchor_parts = cv2.warpAffine(im_anchor_parts, affine, (w, h))
                if k == 0:
                    im_anchor = im_anchor_parts[np.newaxis]
                else:
                    #pdb.set_trace()
                    im_anchor = np.append(im_anchor,im_anchor_parts[np.newaxis],axis=0)

            if j == 0:
                im_ = im_anchor[np.newaxis]
            else:
                im_ = np.append(im_,im_anchor[np.newaxis],axis=0)
        if i == 0:
            im = im_[np.newaxis]
        else:
            im = np.append(im,im_[np.newaxis],axis=0)
    #pdb.set_trace()
    return im
"""
def shift_x_y(coords,H,W,B,image):
    x_shift = tf.reshape(coords[:,:,:,0],[-1,H,W,B])
    y_shift = tf.reshape(coords[:,:,:,1],[-1,H,W,B])
    mag = tf.reshape(coords[:,:,:,2],[-1,H,W,B])

    for h in range(H):
        for w in range(W):
            x_ = x_shift[:,h,w,:] #<tf.Tensor 'strided_slice_4:0' shape=(?, 5) dtype=float32>
            y_ = y_shift[:,h,w,:] #<tf.Tensor 'strided_slice_5:0' shape=(?, 5) dtype=float32>
            mag_ = mag[:,h,w,:] #<tf.Tensor 'strided_slice_6:0' shape=(?, 5) dtype=float32>

            new = tf.Variable(tf.zeros([361],tf.int32))
            for i in range(H):
                for j in range(W):
                    new_x = mag_*i + x_ #<tf.Tensor 'add:0' shape=(?, 5) dtype=float32>
                    new_y = mag_*j + y_ #<tf.Tensor 'add_1:0' shape=(?, 5) dtype=float32>
                    #pdb.set_trace()
                    for k in range(B):
                        #new[tf.transpose(new_x)[k]][tf.transpose(new_y)[k]] = tf.cond(tf.transpose(new_x)[k]>W,lambda:)
                        """
                        if tf.transpose(new_x)[k] > W:
                            pass
                        elif tf.transpose(newy)[k] > H:
                            pass
                        else:
                            new[tf.transpose(new_x)[k]][tf.transpose(new_y)[k]] = image[h][w][k][i][j]
                        """
                        x = tf.cast(tf.transpose(new_x)[k],tf.int32)
                        y = tf.cast(tf.transpose(new_y)[k],tf.int32)
                        pdb.set_trace()
                        try:
                            print("success {0}".format(k))
                        except:
                            pass
                        pdb.set_trace()


    return 0


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
    anchors = mask_anchor(anchors,H)
    anchors=anchors.astype(np.float32)
    #pdb.set_trace()



    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH	     = {}'.format(H))
    print('\tW	     = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [4])
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

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1 + C)]) #<tf.Tensor 'Reshape:0' shape=(?, 19, 19, 5, 85) dtype=float32> x,y,w,h残差
    coords = net_out_reshape[:, :, :, :, :4]
    coords = tf.reshape(coords, [-1, H*W, B, 4])
    #adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2]) #<tf.Tensor 'truediv:0' shape=(?, 361, 5, 2) dtype=float32>
    #adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2])) #<tf.Tensor 'Sqrt:0' shape=(?, 361, 5, 2) dtype=float32>
    #adjusted_coords_x =
    #pdb.set_trace()
    anchors_ = shift_x_y(coords,H,W,B,anchors)
    pdb.set_trace()
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)  #<tf.Tensor 'concat_2:0' shape=(?, 361, 5, 4) dtype=float32>
    pdb.set_trace()
    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1]) #<tf.Tensor 'Reshape_2:0' shape=(?, 361, 5, 1) dtype=float32>

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C]) #<tf.Tensor 'Reshape_3:0' shape=(?, 361, 5, 80) dtype=float32>

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3) #<tf.Tensor 'concat_3:0' shape=(?, 361, 5, 85) dtype=float32>

    wh = tf.pow(coords[:,:,:,2:4], 2) * np.reshape([W, H], [1, 1, 1, 2]) #<tf.Tensor 'mul_23:0' shape=(?, 361, 5, 2) dtype=float32>
    area_pred = wh[:,:,:,0] * wh[:,:,:,1] #<tf.Tensor 'mul_24:0' shape=(?, 361, 5) dtype=float32>
    centers = coords[:,:,:,0:2] #<tf.Tensor 'strided_slice_8:0' shape=(?, 361, 5, 2) dtype=float32>
    floor = centers - (wh * .5) #<tf.Tensor 'sub:0' shape=(?, 361, 5, 2) dtype=float32>
    ceil  = centers + (wh * .5) #<tf.Tensor 'add_2:0' shape=(?, 361, 5, 2) dtype=float32>

    # calculate the intersection areas
    intersect_upleft   = tf.maximum(floor, _upleft) #<tf.Tensor 'Maximum:0' shape=(?, 361, 5, 2) dtype=float32>
    intersect_botright = tf.minimum(ceil , _botright) #<tf.Tensor 'Minimum:0' shape=(?, 361, 5, 2) dtype=float32>
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0) #<tf.Tensor 'Maximum_1:0' shape=(?, 361, 5, 2) dtype=float32>
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1]) #<tf.Tensor 'Mul_27:0' shape=(?, 361, 5) dtype=float32>

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect) #<tf.Tensor 'truediv_3:0' shape=(?, 361, 5) dtype=float32>
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box) #<tf.Tensor 'ToFloat:0' shape=(?, 361, 5) dtype=float32>
    confs = tf.multiply(best_box, _confs) #<tf.Tensor 'Mul_28:0' shape=(?, 361, 5) dtype=float32>
    #pdb.set_trace()
    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3) #<tf.Tensor 'concat_4:0' shape=(?, 361, 5, 4) dtype=float32>
    cooid = scoor * weight_coo #<tf.Tensor 'add_4:0' shape=(?, 361, 5) dtype=float32>
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3) #<tf.Tensor 'concat_5:0' shape=(?, 361, 5, 80) dtype=float32>
    proid = sprob * weight_pro #<tf.Tensor 'mul_32:0' shape=(?, 361, 5, 80) dtype=float32>

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3) #<tf.Tensor 'concat_6:0' shape=(?, 361, 5, 85) dtype=float32>
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3) #<tf.Tensor 'concat_7:0' shape=(?, 361, 5, 85) dtype=float32>

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true, 2)
    pdb.set_trace()
    loss = tf.multiply(loss, wght) #<tf.Tensor 'Mul_34:0' shape=(?, 361, 5, 85) dtype=float32>
    pdb.set_trace()
    loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)]) #<tf.Tensor 'Reshape_4:0' shape=(?, 153425) dtype=float32>
    pdb.set_trace()
    loss = tf.reduce_sum(loss, 1) #<tf.Tensor 'Sum:0' shape=(?,) dtype=float32>
    pdb.set_trace()
    print(iou)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)
