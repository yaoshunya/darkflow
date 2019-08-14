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
def mask_anchor(anchor,H):
    img_x = 1000
    img_y = 1000

    step_x=0
    step_y=0
    S = H
    anchor = np.reshape(anchor,(5,2))
    anchor_size = anchor.shape[0]
    step_size = int(img_x/S)


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
                resize_mask = np.resize(mask_base,(H,H))
                if l == 0:
                    mask = resize_mask[np.newaxis]
                    mask_row = mask_base[np.newaxis]
                else:
                    mask = np.append(mask,resize_mask[np.newaxis],axis=0)
                    mask_row = np.append(mask_row,mask_base[np.newaxis],axis=0)


            step_x += step_size
            if t == 0:
                mask_ = mask[np.newaxis]
                mask__row=mask_row[np.newaxis]
            else:
                mask_ = np.append(mask_,mask[np.newaxis],axis=0)
                mask__row = np.append(mask__row,mask_row[np.newaxis],axis=0)
        print(i)
        step_y += step_size
        if i == 0:
            mask_fi = mask_[np.newaxis]
            mask_fi_row = mask__row[np.newaxis]
        else:
            mask_fi = np.append(mask_fi,mask_[np.newaxis],axis=0)
            mask_fi_row = np.append(mask_fi_row,mask__row[np.newaxis],axis=0)

    with open('data/anchor/anchor.binaryfile','wb') as anc:
        pickle.dump(mask_fi_row,anc,protocol=4)
    return mask_fi


def expit_tensor(x):
	return 1. / (1. + tf.exp(-x))

def shift_x_y(coords,H,W,B,image):

    image = tf.reshape(tf.convert_to_tensor(image,dtype=tf.float32),(H*W,B,H,W))
    n_fc = 6

    h_trans = spatial_transformer_network(image, coords)

    return h_trans

#https://stackoverflow.com/questions/42252040/how-to-translateor-shift-images-in-tensorflow
#https://zhengtq.github.io/2018/12/20/tf-tur-perspective-transform/

def condition(i,N,theta,input_fmap,l_):
    return i < N

def update(i,N,theta,input_fmap,l_):
    #pdb.set_trace()
    batch_grids = affine_grid_generator(19, 19, theta[i])

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
    l_ = l_.write(i,out_fmap)
    return i+1,N,theta,tf.transpose(out_fmap,[0,3,1,2]),l_

def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):

    x_shift = tf.transpose(theta)[0]
    y_shift = tf.transpose(theta)[1]
    z_rotate = tf.transpose(theta)[2]
    sin = tf.math.sin(z_rotate)
    cos = tf.math.cos(z_rotate)
    batch_size = tf.shape(theta)[0]
    new_theta=tf.transpose(tf.stack([cos,tf.math.negative(sin),x_shift,sin,cos,y_shift]))
    new_theta = tf.reshape(new_theta,[batch_size,361,5,2,3])
    #pdb.set_trace()
    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]
    #pdb.set_trace()

    # reshape theta to (B, 2, 3)
    #theta = tf.reshape(theta, [B, 2, 3])
    out_list = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
    init_val = (0,batch_size,theta,input_fmap,out_list)
    #pdb.set_trace()

    out_fmap = tf.while_loop(cond=condition,body=update,loop_vars=init_val)
    out_stack = out_fmap[4].stack()
    return out_stack

def get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, theta):
    
    num_batch = tf.shape(theta)[0]
    
    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    #pdb.set_trace()
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 5, height, width])
    #pdb.set_trace()
    return batch_grids


def bilinear_sampler(img, x, y):
    #pdb.set_trace()
    img = tf.transpose(img,[0,2,3,1])
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def normalize_(x):

    x = tf.reshape(x,(200,361,361,5))
    x_max = tf.tile(tf.expand_dims(tf.argmax(x,2),2),[1,1,361,1])
    x_min = tf.tile(tf.expand_dims(tf.argmin(x,2),2),[1,1,361,1])
    x = tf.math.divide(tf.subtract(tf.cast(x,tf.int64),x_min),tf.subtract(x_max,x_min))

    return x

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

    

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH	     = {}'.format(H))
    print('\tW	     = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]
    size3 = [None, HW, B, H, W]
    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    #pdb.set_trace()
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [3])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size3)
    #_upleft = tf.placeholder(tf.float32, size2 + [2])
    #_botright = tf.placeholder(tf.float32, size2 + [2])

    self.placeholders = {
	'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
	'areas':_areas
    }
    #pdb.set_trace()
    # Extract the coordinate prediction from net.out
    #pdb.set_trace()
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (3 + 1 + C)]) #<tf.Tensor 'Reshape:0' shape=(?, 19, 19, 5, 85) dtype=float32> x,y,w,h残差
    coords = net_out_reshape[:, :, :, :, :3]
    coords = tf.reshape(coords, [-1, H*W, B, 3])
    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 3])
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])
    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 4:])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

    area_pred = shift_x_y(coords,H,W,B,anchors)

    _areas = tf.transpose(_areas,(0,1,3,4,2))
    #pdb.set_trace()
    batch_size = tf.shape(coords)[0]
    intersect = tf.math.multiply(tf.cast(area_pred,tf.float64),tf.cast(_areas,tf.float64))
    #pdb.set_trace()
    intersect = tf.reduce_sum(tf.math.sign(tf.reshape(intersect,(batch_size,361,361,5))),2) #<tf.Tensor 'Sum:0' shape=(?, 361, 5) dtype=float64>
    area_pred = tf.reduce_sum(tf.math.sign(tf.reshape(area_pred,(batch_size,361,361,5))),2) #<tf.Tensor 'Sum_1:0' shape=(?, 361, 5) dtype=float32>
    _areas= tf.reduce_sum(tf.math.sign(tf.reshape(_areas,(batch_size,361,361,5))),2) #<tf.Tensor 'Sum_2:0' shape=(?, 361, 5) dtype=float32>


    iou = tf.math.divide(intersect,(tf.cast(_areas+area_pred,tf.float64)-intersect)+1e-10) #<tf.Tensor 'truediv:0' shape=(?, 361, 5) dtype=float64>
    adjusted_net_out = tf.concat([coords, adjusted_c, adjusted_prob], 3)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(3 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)
    #true = tf.concat([tf.expand_dims(confs,3),_probs],3)
    #wght = tf.concat([tf.expand_dims(coind,3),proid],3)
    """
    intersect_num = tf.math.count_nonzero(tf.reshape(intersect,(200,361,361,5)),2)
    area_pred_num = tf.math.count_nonzero(tf.reshape(area_pred,(200,361,361,5)),2)
    _areas_num = tf.math.count_nonzero(tf.reshape(_areas,(200,361,361,5)),2)
    """
    #iou = tf.reshape(iou,[-1,H*W*B])
    #loss = 1-tf.reduce_sum(iou,1)

    iou_loss = tf.reduce_sum(tf.reduce_sum(iou,1),1)
    print('Building {} loss'.format(m['model']))
    #pdb.set_trace()
    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, H*W*B*(3 + 1 + C)])
    loss = tf.reduce_sum(loss, 1) - tf.cast(iou_loss,tf.float32)*1000
    #pdb.set_trace()
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)
