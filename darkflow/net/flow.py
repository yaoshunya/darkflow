import os
import time
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool
from tensorflow .python import debug as tf_debug
import pdb
import cv2
from ..cython_utils.cy_yolo2_findboxes import box_constructor
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from statistics import mean
import seaborn as sns
import random
train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def detect_most_near(pre,annotations,image_name):
    for ann in annotations:
        if ann[0] == image_name:
            error_list = list()
            iou_list = list()
            if len(ann[1]) == 0:
                return np.zeros((1000,1000,3)),100
                
            for i in range(len(ann[1])):
                pre_ = np.zeros((1000,1000))
                pre_[pre[:,:,0]>0] = 1
                pre_ = np.reshape(pre_,[-1])
                X = np.zeros((1000,1000))
                X[ann[1][i][1]] = 1
                cm = confusion_matrix(np.reshape(X,[-1]),pre_)

                TP = cm[0][0]
                FP = cm[0][1]
                FN = cm[1][0]
                TP = cm[1][1]
                iou = TP/(TP+FP+FN)
                """
                my_list_ann = []
                my_list_pre = []
                #pdb.set_trace()
                pre_len_ = len(np.where(pre>0)[0])
                ann_len_ = len(ann[1][0][1][0])
                pre_coords = np.where(pre>0)
                for k in range(30):
                    x = random.randint(0,ann_len_-1)
                    y = random.randint(0,pre_len_-1)
                    my_list_ann.append(x)
                    my_list_pre.append(y)

                ann_stack = np.vstack((ann[1][0][1][0][my_list_ann],ann[1][0][1][1][my_list_ann]))
                pre_stack = np.vstack((pre_coords[0][y],pre_coords[1][y]))

                dcpoints = ann_stack - pre_stack
                d = np.linalg.norm(dcpoints,axis=0)
                error = sum(d)
                if error < 5000:
                    
                    #pdb.set_trace()
                else:
                    iou = 0
                #error_list.append(error)
                """
                iou_list.append(iou)
            max_index = np.argmax(np.array(iou_list))
            #pdb.set_trace()
            X = np.zeros((1000,1000))
            X[ann[1][max_index][1]] = 255
            X = np.tile(np.transpose(X[np.newaxis],[1,2,0]),[1,1,3])
            #pdb.set_trace()
            return X,iou_list[max_index]
    return np.zeros((1000,1000,3)),0

def make_result(out,this_batch):

    batch_size = len(this_batch)

    with open('data/ann_anchor_data/mask_anchor_k.pickle', 'rb') as f:
        anchor = pickle.load(f)
    with open('data/ann_anchor_data/max_min_k.pickle','rb') as f:
        X = pickle.load(f)

    with open('data/ann_anchor_data/ann_coords_4.pickle','rb') as f:
        annotations = pickle.load(f)


    #t_0_max = np.array(X[0])
    t_0_max = np.array(547.0)
    t_0_min = np.array(X[1])
    #t_1_max = np.array(X[2])
    t_1_max = np.array(486.0)
    t_1_min = np.array(-64.0)
    #t_1_min = np.array(X[3])
    #pdb.set_trace()
    iou_return = list()
    precision_return = list()
    recall_return = list()
    iou_label =  [[] for i in range(6)]
    precision_label = [[] for i in range(6)]
    recall_label = [[] for i in range(6)]
    T_0_list = list()
    T_1_list = list()
    R_list = list()
    for i in range(batch_size):
        #pdb.set_trace()
        out_now = np.transpose(np.reshape(out[i],[361,5,6]),[2,0,1])
        image_name = this_batch[i]

        out_conf = np.reshape(out_now[3],[-1])

        confidence = (1/(1+np.exp(-out_conf)))
        #trast_conf = np.where(confidence>0.1)
        K = 3
        X = np.argpartition(-confidence,K)[:K]
        y = confidence[X]
        indices = np.argsort(-y)
        trast_conf = X[indices]
        #print(trast_conf[0])
        pre = np.zeros((1000,1000))
        #pdb.set_trace()
        if len(trast_conf) == 0:
            continue
        for j in range(len(trast_conf)):
            #pdb.set_trace()
            """
            if confidence[trast_conf[j]] < 0.4:
                continue
            """
            print(confidence[trast_conf[j]])
            R = np.reshape(out_now[0],[-1])[trast_conf[j]]
            #print(R)
            T_0 = np.dot(np.divide(np.reshape(out_now[1],[-1])[trast_conf[j]]+1,2),t_0_max-t_0_min)+t_0_min
            T_1 = np.dot(np.divide(np.reshape(out_now[2],[-1])[trast_conf[j]]+1,2),t_1_max-t_1_min)+t_1_min
            
            anchor_now = np.reshape(anchor,[1805,1000,1000])[trast_conf[j]]
            
            affine = np.array([[1,0,T_1],[0,1,T_0]])
            
            pre=cv2.warpAffine(anchor_now, affine, (1000,1000))
            
            affine = cv2.getRotationMatrix2D((0,0),R,1.0)
            
            pre=cv2.warpAffine(pre, affine, (1000,1000))
            
            
            
            where_pre = np.where(pre>0)
            x_ = mean(where_pre[0])/2
            y_ = mean(where_pre[1])/2
            label = where_pre_label(x_,y_)
            
            pre = np.tile(np.transpose(pre[np.newaxis],[1,2,0]),[1,1,3])

            pre[:,:,1] = 0
            pre[:,:,2]/255
            pre[:,:,2]*200
            #pdb.set_trace()
            ann,iou_parts = detect_most_near(pre,annotations,image_name)
            if iou_parts == 100:
                continue
            T_0_list.append(T_0)
            T_1_list.append(T_1)
            R_list.append(R)

            mask = 1*(cv2.resize(ann[:,:,0],(1000,1000))>0)
            prediction = 1*(pre[:,:,0]>0)

            precision_parts = precision_score(np.reshape(mask,[-1]),np.reshape(prediction,[-1]))
            recall_parts = recall_score(np.reshape(mask,[-1]),np.reshape(prediction,[-1]))

            iou_return.append(iou_parts)
            precision_return.append(precision_parts)
            recall_return.append(recall_parts)
            iou_label[label-1].append(iou_parts)
            precision_label[label-1].append(precision_parts)
            recall_label[label-1].append(recall_parts)

            imgcv = cv2.imread(os.path.join('data/VOC2012/sphere_test',this_batch[i]))
            prediction = cv2.addWeighted(np.asarray(imgcv,np.float64),0.7,np.asarray(pre,np.float64),0.3,0)
            prediction = cv2.addWeighted(np.asarray(prediction,np.float64),0.6,np.asarray(ann,np.float64),0.4,0)
            cv2.imwrite('data/out_test/test_image_{0}_{1}.png'.format(this_batch[i][:6],j),prediction)
   
    return iou_return,precision_return,recall_return,T_0_list,T_1_list,R_list,iou_label,precision_label,recall_label


import math

def where_pre_label(x,y):
    if x < 500:
        if y <400:
            label = 1
        elif 400 <= y < 600:
            label = 2
        else:
            label = 3
    else:
        if y <400:
            label = 4
        elif 400 <= y < 600:
            label = 5
        else:
            label = 6
    return label



def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))

    iou_ = list()
    precision_ = list()
    recall_ = list()
    T_0_ = list()
    T_1_ = list()
    R_ = list()

    iou_label_all =  [[] for i in range(6)]
    precision_label_all = [[] for i in range(6)]
    recall_label_all = [[] for i in range(6)]
    i=0
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))
        print(i)
        i = i+1
        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()

        iou,precision,recall,T_0,T_1,R,iou_label,precision_label,recall_label = make_result(out,this_batch)
        iou_.extend(iou)
        precision_.extend(precision)
        recall_.extend(recall)
        T_0_.extend(T_0)
        T_1_.extend(T_1)
        R_.extend(R)

        [iou_label_all[i].extend(iou_label[i]) for i in range(6)]
        [precision_label_all[i].extend(precision_label[i]) for i in range(6)]
        [recall_label_all[i].extend(recall_label[i]) for i in range(6)]
        if i == 30:
            break
    #[iou_label_mean.append(mean(iou_label_all[i])) for i in range(6)]
    #[precision_label_mean.append(mean(precision_label_all[i])) for i in range(6)]
    #[recall_label_mean.append(mean(recall_label_all[i])) for i in range(6)]
    
    with open('data/out_data/iou_label.pickle',mode = 'wb') as f:
            pickle.dump(iou_label,f) 
    with open('data/out_data/precision_label.pickle',mode='wb') as f:
            pickle.dump(precision_label,f)
    with open('data/out_data/recall_label.pickle',mode='wb') as f:
            pickle.dump(recall_label,f)
    with open('data/out_data/R.pickle',mode = 'wb') as f:
            pickle.dump(R_,f)
    with open('data/out_data/T_0.pickle',mode='wb') as f:
            pickle.dump(T_0,f)
    with open('data/out_data/T_1.pickle',mode='wb') as f:
            pickle.dump(T_1,f)
    
    plt.hist(np.array(T_0_),color='blue')
    plt.savefig('../GoogleDrive/T_0.png')
    plt.clf()
    plt.hist(np.array(T_1_),color='blue')
    plt.savefig('../GoogleDrive/T_1.png')
    plt.clf()
    plt.hist(np.array(R_),color='blue')
    plt.savefig('../GoogleDrive/R.png')
    plt.clf()
    
    x_idx = [i+1 for i in range(6)]
    
    plt_all = [iou_label_all,precision_label_all,recall_label_all]
    label_all = ['iou','precision','recall']
    """
    for i in range(len(plt_all)):   
        plt.plot(x_idx,plt_all[i],'o',color='blue')
        plt.xlabel("label")
        plt.ylabel("{0}".format(label_all[i]))
        plt.savefig('../GoogleDrive/{0}.png'.format(label_all[i]))
        plt.clf()
    """
    """
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
    """
        #stop = time.time(); last = stop - start

    """
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
    """
    print("iou:{0}".format(np.nanmean(iou_)))
    print("precision:{0}".format(np.nanmean(precision_)))
    print("recall:{0}".format(np.nanmean(recall_)))
    #pdb.set_trace()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt:
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    #pdb.set_trace()
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    #pdb.set_trace()
    loss_op = self.framework.loss
    step_plot = np.array([])
    loss_plot = np.array([])
    loss_plot_ = np.array([])
    s = 0
    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key]
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        fetched = self.sess.run(fetches, feed_dict)

        loss = fetched[1]


        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)
        #pdb.set_trace()
        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))

        #step_plot = np.append(step_plot,step_now)
        loss_plot = np.append(loss_plot,loss)

        if step_now%50 == 0:
            step_plot = np.append(step_plot,step_now)
            #loss_plot = np.append(loss_plot,loss)
            loss_plot_ = np.append(loss_plot_,np.mean(loss_plot))
            plt.plot(step_plot,loss_plot_)

            plt.xlabel("step")
            plt.ylabel("loss")
            plt.savefig('data/out_test/figure.png')
            plt.clf()
            loss_plot = np.array([])
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)


def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo
