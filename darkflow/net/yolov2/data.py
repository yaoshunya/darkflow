from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from ..yolo.predict import preprocess
from ..yolo.data import shuffle
from copy import deepcopy
import pickle
import numpy as np
import os
import pdb
import cv2
def _batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's
    input & loss layer correspond to this chunk
    """
    
    #pdb.set_trace()
    meta = self.meta
    labels = meta['labels']

    H, W, _ = meta['out_size']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']

    # preprocess
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(self.FLAGS.dataset, jpg)
    #pdb.set_trace()
    img = self.preprocess(path, allobj)
    #img = cv2.imread(path)
    #img = slef.
    #i = 0
    # Calculate regression target
    
    cellx = 1. * 1000 / W
    celly = 1. * 1000 / H
    for obj in allobj:
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax
        centery = .5*(obj[2]+obj[4]) #ymin, ymax
        cx = centerx / cellx
        cy = centery / celly
        if cx >= W or cy >= H: return None, None
        obj[3] = float(obj[3]-obj[1]) / 1000
        obj[4] = float(obj[4]-obj[2]) / 1000
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery
        obj += [int(np.floor(cy) * W + np.floor(cx))]
        #pdb.set_trace()
    

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([H*W,B,C])
    confs = np.zeros([H*W,B])
    coord = np.zeros([H*W,B,4])
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,H*W])
    
    for obj in allobj:
        probs[obj[6], :, :] = [[0.]*C] * B
        probs[obj[6], :, labels.index(obj[0])] = 1.
        proid[obj[6], :, :] = [[1.]*C] * B
        coord[obj[6], :, :] = [obj[1:5]] * B
        prear[obj[6],:] = obj[5] # mask
        #prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * H # yup
        #prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * W # xright
        #prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * H # ybot
        confs[obj[6], :] = [1.] * B
    
    
    # Finalise the placeholders' values
    """
    flag = 0
    for obj in allobj:
        flag = 1
        if i == 0:
            areas = obj[1][np.newaxis]
            #pdb.set_trace()
        else:
            areas = np.append(areas,obj[1][np.newaxis],axis = 0)
        i = i + 1
    if flag == 1:
        areas = areas.T
    
    if flag == 0:
        upleft   = np.expand_dims(prear[:,0:2], 1)
        botright = np.expand_dims(prear[:,2:4], 1)
        wh = botright - upleft;
        area = wh[:,:,0] * wh[:,:,1]
        areas = np.concatenate([area] * B, 1)
        #pdb.set_trace()
    """
    areas = np.tile(prear[np.newaxis].T,(1,1,5))
    #pdb.set_trace()
    areas = np.reshape(areas,(361,5,19,19))
    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'areas': areas
    }
    #pdb.set_trace()
    return inp_feed_val, loss_feed_val
