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
                resize_mask = np.resize(mask_base,(H,H))
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
    x_shift = tf.reshape(coords[:,:,:,0],[-1,H,W,B])#<tf.Tensor 'Reshape_2:0' shape=(?, 19, 19, 5) dtype=float32>
    y_shift = tf.reshape(coords[:,:,:,1],[-1,H,W,B])
    theta = tf.reshape(coords[:,:,:,2],[-1,H,W,B])

    for h in range(H):
        for w in range(W):
            x_ = x_shift[:,h,w,:] #<tf.Tensor 'strided_slice_4:0' shape=(?, 5) dtype=float32>
            y_ = y_shift[:,h,w,:] #<tf.Tensor 'strided_slice_5:0' shape=(?, 5) dtype=float32>
            theta_ = theta[:,h,w,:] #<tf.Tensor 'strided_slice_6:0' shape=(?, 5) dtype=float32>
            new_x = tf.math.cos(theta_)*h-tf.math.sin(theta_)*h+x_ #<tf.Tensor 'add:0' shape=(?, 5) dtype=float32>
            new_y = tf.math.sin(theta_)*w-tf.math.cos(theta_)*w+y_ #<tf.Tensor 'add_1:0' shape=(?, 5) dtype=float32>
            for k in range(B):
                x = tf.cast(tf.transpose(new_x)[k],tf.float32)
                y = tf.cast(tf.transpose(new_y)[k],tf.float32)
                imgs = image[h][w][k]
                im = my_img_translate(imgs, x,y)
                pdb.set_trace()
                im = tf.cast(im,tf.int32)
                im = tf.expand_dims(tf.reshape(im,[H,W]),0)
                if k == 0:
                    im_k = im
                else:
                    im_k = tf.concat([im_k,im],0)
            im_k = tf.expand_dims(im_k,0)
            if h == 0 and w == 0:
                im_w=im_k
            else:
                im_w=tf.concat([im_w,im_k],0)
                print(h)

    return im_w
"""
#https://stackoverflow.com/questions/42252040/how-to-translateor-shift-images-in-tensorflow
# Tensorflow image translation op
# images:        A tensor of shape (num_images, num_rows, num_columns, num_channels) (NHWC),
#                (num_rows, num_columns, num_channels) (HWC), or (num_rows, num_columns) (HW).
# tx:            The translation in the x direction.
# ty:            The translation in the y direction.
# interpolation: If x or y are not integers, interpolation comes into play. Options are 'NEAREST' or 'BILINEAR'
def tf_image_translate(images, tx, ty, interpolation='NEAREST'):
    # got these parameters from solving the equations for pixel translations
    # on https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform
    transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
    return tf.contrib.image.transform(images, transforms, interpolation)

#https://zhengtq.github.io/2018/12/20/tf-tur-perspective-transform/
def transform_perspective(image):
    def x_y_1():
        x = tf.random_uniform([], minval=-0.3, maxval=-0.15)
        y = tf.random_uniform([], minval=-0.3, maxval=-0.15)
        return x, y

    def x_y_2():
        x = tf.random_uniform([], minval=0.15, maxval=0.3)
        y = tf.random_uniform([], minval=0.15, maxval=0.3)
        return x, y

    def trans(image):
        ran = tf.random_uniform([])
        x = tf.random_uniform([], minval=-0.3, maxval=0.3)
        x_com = tf.random_uniform([], minval=1-x-0.1, maxval=1-x+0.1)

        y = tf.random_uniform([], minval=-0.3, maxval=0.3)
        y_com = tf.random_uniform([], minval=1-y-0.1, maxval=1-y+0.1)

        transforms =  [x_com, x,0,y,y_com,0,0.00,0]
        pdb.set_trace()
        ran = tf.random_uniform([])
        image = tf.cond(ran<0.5, lambda:tf.contrib.image.transform(image,transforms,interpolation='NEAREST', name=None),
                lambda:tf.contrib.image.transform(image,transforms,interpolation='BILINEAR', name=None))
        return image

    ran = tf.random_uniform([])
    image = tf.cond(ran<1, lambda: trans(image), lambda:image)

    return image
"""
def my_img_translate(imgs, x,y):
    # Interpolation model has to be fixed due to limitations of tf.custom_gradient
    interpolation = 'NEAREST'
    imgs = imgs[np.newaxis,:,:,np.newaxis]
    imgs = tf.convert_to_tensor(imgs,dtype=tf.float32)
    x=tf.expand_dims(x,1)
    y=tf.expand_dims(y,1)
    translates = tf.concat([x,y],1)
    #pdb.set_trace()
    imgs_translated = tf.contrib.image.translate(imgs, translates, interpolation=interpolation)
    #pdb.set_trace()
    def grad(img_translated_grads):
        translates_x = translates[:, 0] #<tf.Tensor 'gradients/IdentityN_grad/strided_slice:0' shape=(?,) dtype=float32>
        translates_y = translates[:, 1] #<tf.Tensor 'gradients/IdentityN_grad/strided_slice_1:0' shape=(?,) dtype=float32>
        translates_zero = tf.zeros_like(translates_x) #<tf.Tensor 'gradients/IdentityN_grad/zeros_like:0' shape=(?,) dtype=float32>
        # X gradients
        imgs_x_grad = (imgs[:, :, :-2] - imgs[:, :, 2:]) / 2 #<tf.Tensor 'gradients/IdentityN_grad/truediv:0' shape=(?, ?, ?, ?) dtype=float32>
        imgs_x_grad = tf.concat([(imgs[:, :, :1] - imgs[:, :, 1:2]),
                                 imgs_x_grad,
                                 (imgs[:, :, -2:-1] - imgs[:, :, -1:])], axis=2) #<tf.Tensor 'gradients/IdentityN_grad/truediv:0' shape=(?, ?, ?, ?) dtype=float32>
        imgs_x_grad_translated = tf.contrib.image.translate(
            imgs_x_grad, tf.stack([translates_x, translates_zero], axis=1),
            interpolation=interpolation)
        translates_x_grad = tf.reduce_sum(img_translated_grads * imgs_x_grad_translated, axis=(1, 2, 3)) #
        # Y gradients
        imgs_y_grad = (imgs[:, :-2] - imgs[:, 2:]) / 2
        imgs_y_grad = tf.concat([(imgs[:, :1] - imgs[:, 1:2]),
                                 imgs_y_grad,
                                 (imgs[:, -2:-1] - imgs[:, -1:])], axis=1)
        imgs_y_grad_translated = tf.contrib.image.translate(
            imgs_y_grad, tf.stack([translates_zero, translates_y], axis=1),
            interpolation=interpolation)
        translates_y_grad = tf.reduce_sum(img_translated_grads * imgs_y_grad_translated, axis=(1, 2, 3))
        # Complete gradient
        translates_grad = tf.stack([translates_x_grad, translates_y_grad], axis=1) #<tf.Tensor 'gradients/IdentityN_grad/stack_2:0' shape=(?, 2) dtype=float32>

        return None, translates_grad
    gr = grad(imgs_translated)
    return imgs_translated,gr


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
    size3 = [None, HW, B, H, W]
    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size3)
    _coord = tf.placeholder(tf.float32, size2 + [4])
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

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (3 + 1 + C)]) #<tf.Tensor 'Reshape:0' shape=(?, 19, 19, 5, 85) dtype=float32> x,y,w,h残差
    coords = net_out_reshape[:, :, :, :, :3]
    coords = tf.reshape(coords, [-1, H*W, B, 3])
    #adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2]) #<tf.Tensor 'truediv:0' shape=(?, 361, 5, 2) dtype=float32>
    #adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2])) #<tf.Tensor 'Sqrt:0' shape=(?, 361, 5, 2) dtype=float32>
    #adjusted_coords_x =
    #pdb.set_trace()
    area_pred = shift_x_y(coords,H,W,B,anchors)
    area_pred = tf.transpose(area_pred,(0,2,3,1))
    max = tf.cast(tf.tile(tf.expand_dims(tf.argmax(area_pred,3),3),[1,1,1,5]),tf.int32)
    min = tf.cast(tf.tile(tf.expand_dims(tf.argmin(area_pred,3),3),[1,1,1,5]),tf.int32)
    area_pred = tf.reshape(tf.math.divide(tf.math.subtract(area_pred,min),tf.math.subtract(max,min)),[-1,HW,H,W,B])
    _areas = tf.transpose(_areas,(0,1,3,4,2))
    intersect = tf.math.multiply(tf.cast(area_pred,tf.float64),tf.cast(_areas,tf.float64))
    area_pred = tf.cast(area_pred,tf.float32)
    intersect = tf.cast(intersect,tf.float32)
    iou = tf.truediv(intersect, _areas + area_pred - intersect)
    loss = 1-tf.reshape(iou,[-1])
    pdb.set_trace()
    #coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)  #<tf.Tensor 'concat_2:0' shape=(?, 361, 5, 4) dtype=float32>
    #pdb.set_trace()
    #adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
    #adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1]) #<tf.Tensor 'Reshape_2:0' shape=(?, 361, 5, 1) dtype=float32>

    #adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
    #adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C]) #<tf.Tensor 'Reshape_3:0' shape=(?, 361, 5, 80) dtype=float32>

    #adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3) #<tf.Tensor 'concat_3:0' shape=(?, 361, 5, 85) dtype=float32>

    #wh = tf.pow(coords[:,:,:,2:4], 2) * np.reshape([W, H], [1, 1, 1, 2]) #<tf.Tensor 'mul_23:0' shape=(?, 361, 5, 2) dtype=float32>
    #area_pred = wh[:,:,:,0] * wh[:,:,:,1] #<tf.Tensor 'mul_24:0' shape=(?, 361, 5) dtype=float32>
    #centers = coords[:,:,:,0:2] #<tf.Tensor 'strided_slice_8:0' shape=(?, 361, 5, 2) dtype=float32>
    #floor = centers - (wh * .5) #<tf.Tensor 'sub:0' shape=(?, 361, 5, 2) dtype=float32>
    #ceil  = centers + (wh * .5) #<tf.Tensor 'add_2:0' shape=(?, 361, 5, 2) dtype=float32>

    # calculate the intersection areas
    #intersect_upleft   = tf.maximum(floor, _upleft) #<tf.Tensor 'Maximum:0' shape=(?, 361, 5, 2) dtype=float32>
    #intersect_botright = tf.minimum(ceil , _botright) #<tf.Tensor 'Minimum:0' shape=(?, 361, 5, 2) dtype=float32>
    #intersect_wh = intersect_botright - intersect_upleft
    #intersect_wh = tf.maximum(intersect_wh, 0.0) #<tf.Tensor 'Maximum_1:0' shape=(?, 361, 5, 2) dtype=float32>
    #intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1]) #<tf.Tensor 'Mul_27:0' shape=(?, 361, 5) dtype=float32>

    # calculate the best IOU, set 0.0 confidence for worse boxes
    #iou = tf.truediv(intersect, _areas + area_pred - intersect) #<tf.Tensor 'truediv_3:0' shape=(?, 361, 5) dtype=float32>
    #best_box = tf.equal(iou, tf.reduce_max(iou, [4], True))
    #best_box = tf.cast(best_box,tf.float32) #<tf.Tensor 'ToFloat:0' shape=(?, 361, 5) dtype=float32>
    #confs = tf.multiply(best_box, tf.transpose(_confs,[0,1,3,4,2])) #<tf.Tensor 'Mul_28:0' shape=(?, 361, 5) dtype=float32>
    #pdb.set_trace()
    # take care of the weight terms
    #conid = snoob * (1. - confs) + sconf * confs
    #weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3) #<tf.Tensor 'concat_4:0' shape=(?, 361, 5, 4) dtype=float32>
    #cooid = scoor * weight_coo #<tf.Tensor 'add_4:0' shape=(?, 361, 5) dtype=float32>
    #weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3) #<tf.Tensor 'concat_5:0' shape=(?, 361, 5, 80) dtype=float32>
    #proid = sprob * weight_pro #<tf.Tensor 'mul_32:0' shape=(?, 361, 5, 80) dtype=float32>

    #self.fetch += [_probs, confs, conid, cooid, proid]
    #true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3) #<tf.Tensor 'concat_6:0' shape=(?, 361, 5, 85) dtype=float32>
    #wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3) #<tf.Tensor 'concat_7:0' shape=(?, 361, 5, 85) dtype=float32>

    print('Building {} loss'.format(m['model']))
    #loss = tf.pow(adjusted_net_out - true, 2)
    #pdb.set_trace()
    #loss = tf.multiply(loss, wght) #<tf.Tensor 'Mul_34:0' shape=(?, 361, 5, 85) dtype=float32>
    #pdb.set_trace()
    #loss = tf.reshape(loss, [-1, H*W*B*(4 + 1 + C)]) #<tf.Tensor 'Reshape_4:0' shape=(?, 153425) dtype=float32>
    #pdb.set_trace()
    #loss = tf.reduce_sum(loss, 1) #<tf.Tensor 'Sum:0' shape=(?,) dtype=float32>
    #pdb.set_trace()
    #print(iou)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)
