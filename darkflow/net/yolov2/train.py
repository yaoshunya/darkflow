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
                else:
                    mask = np.append(mask,resize_mask[np.newaxis],axis=0)
                print(l)

            step_x += step_size
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

def shift_x_y(coords,H,W,B,image):

    image = tf.reshape(tf.convert_to_tensor(image,dtype=tf.float32),(H*W,B,H,W))
    n_fc = 6

    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32').flatten()
    W_fc1 = tf.Variable(tf.zeros([H*W*(H*W), n_fc]), name='W_fc1')
    b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
    h_fc1 = tf.matmul(tf.zeros([B, H*W*(H*W)]), W_fc1) + b_fc1
    h_trans = spatial_transformer_network(image, coords)
"""
    init_HW=(0,image,shift_image)
    shift_image = tf.while_loop(cond=condition_HW,body=body_HW,loop_vars=init_HW)
def condition_HW(i,image,shift_image):
    return i < 361
def body_HW(i,image,shift_image):
    init_B = (i,0,image,shift_image)
    return tf.while_loop(cond=condition_B,body=body_B,loop_vars=init_B)
def condition_B(i,k,image,shift_image):
    return k < 5
def body_B(i,k,image,shift_image):
    img_B = image[i][k]
    x = tf.where(tf.equal(img_B,255))
    return 0

def condition(zero_matrix):
    return tf.Variable(zero_matrix)

def shift_x_y_(coords,H,W,B,image):
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
                #pdb.set_trace()
                im_gra = im[1][1]
                im = im[0]
                #pdb.set_trace()
                im = tf.cast(im,tf.int32)
                im = tf.expand_dims(tf.reshape(im,[H,W]),0)

                im_gra = tf.expand_dims(im_gra,0)
                if k == 0:
                    im_k = im
                    im_gra_k = im_gra
                else:
                    #pdb.set_trace()
                    im_k = tf.concat([im_k,im],0)
                    im_gra_k = tf.concat([im_gra_k,im_gra],0)
            #pdb.set_trace()
            im_gra_k = tf.expand_dims(im_gra_k,0)
            im_k = tf.expand_dims(im_k,0)
            if h==0 and w==0:
                im_w=im_k
                im_gra_w=im_gra_k
            else:
                im_w=tf.concat([im_w,im_k],0)
                im_gra_w=tf.concat([im_gra_w,im_gra_k],0)
                print(h)

    return im_w,im_gra_w


#https://stackoverflow.com/questions/42252040/how-to-translateor-shift-images-in-tensorflow
#https://zhengtq.github.io/2018/12/20/tf-tur-perspective-transform/

def my_img_translate(imgs, x,y):
    # Interpolation model has to be fixed due to limitations of tf.custom_gradient
    interpolation = 'NEAREST'
    imgs = imgs[np.newaxis,:,:,np.newaxis]
    imgs = tf.convert_to_tensor(imgs,dtype=tf.float32)
    t = tf.placeholder(tf.float32,shape=[None,None,None,None])
    imgs = t+imgs
    x=tf.expand_dims(x,1)
    y=tf.expand_dims(y,1)
    translates = tf.concat([x,y],1)

    imgs_translated = tf.contrib.image.translate(imgs, translates, interpolation=interpolation)

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
"""
"""
def return_image_gra(imgs,H,W):
    for i in range(H*W):

        dy_parts,dx_parts = tf.image.image_gradients(tf.transpose(imgs,[1,0,2,3,4])[i])


        dy_parts = tf.expand_dims(dy_parts,0)
        dx_parts = tf.expand_dims(dx_parts,0)

        if i == 0:
            dy = dy_parts
            dx = dx_parts
        else:
            dy = tf.concat([dy,dy_parts],0)
            dx = tf.concat([dx,dx_parts],0)
    return tf.transpose(dy,[1,0,2,3,4]),tf.transpose(dx,[1,0,2,3,4])
"""
def condition(i,N,theta,input_fmap):
    return i < N

def update(i,N,theta,input_fmap):
    batch_grids = affine_grid_generator(19, 19, theta[i])

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]
    #pdb.set_trace()
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
    pdb.set_trace()
    return 0




def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):

    x_shift = tf.transpose(theta)[0]
    y_shift = tf.transpose(theta)[1]
    z_rotate = tf.transpose(theta)[2]
    sin = tf.math.sin(z_rotate)
    cos = tf.math.cos(z_rotate)

    new_theta=tf.transpose(tf.stack([cos,tf.math.negative(sin),x_shift,sin,cos,y_shift]))
    new_theta = tf.reshape(new_theta,[200,361,5,2,3])
    #pdb.set_trace()
    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]
    #pdb.set_trace()

    # reshape theta to (B, 2, 3)
    #theta = tf.reshape(theta, [B, 2, 3])
    init_val = (0,200,theta,input_fmap)
    out_fmap = tf.while_loop(cond=condition,body=update,loop_vars=init_val)
    return out_fmap
"""
    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)


    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
"""



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

    return batch_grids


def bilinear_sampler(img, x, y):
    pdb.set_trace()
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

    size1 = [200, HW, B, C]
    size2 = [200, HW, B]
    size3 = [200, HW, B, H, W]
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

    area_pred,gr = shift_x_y(coords,H,W,B,anchors)

    area_pred = tf.transpose(area_pred,(0,2,3,1))
    max = tf.cast(tf.tile(tf.expand_dims(tf.argmax(area_pred,3),3),[1,1,1,5]),tf.int32)
    min = tf.cast(tf.tile(tf.expand_dims(tf.argmin(area_pred,3),3),[1,1,1,5]),tf.int32)
    area_pred = tf.reshape(tf.math.divide(tf.math.subtract(area_pred,min),tf.math.subtract(max,min)),[-1,HW,H,W,B])
    _areas = tf.transpose(_areas,(0,1,3,4,2))
    intersect = tf.math.multiply(tf.cast(area_pred,tf.float64),tf.cast(_areas,tf.float64))
    area_pred = tf.cast(area_pred,tf.float32)
    intersect = tf.cast(intersect,tf.float32)
    iou = tf.truediv(intersect, _areas + area_pred - intersect) #<tf.Tensor 'truediv_3611:0' shape=(?, 361, 19, 19, 5) dtype=float32>


    print('Building {} loss'.format(m['model']))

    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)
