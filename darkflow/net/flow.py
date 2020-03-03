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

def detect_most_near(pre,ann):
#予測と真値を比較し、最もIoUが高いものを真値として採用する
        if len(ann[1]) == 0:
            #予測しているにも関わらず、その画像に物体が存在しない場合は、iou_list=1000とする
            return np.zeros((1000,1000,3)),1000,ann[0]
        iou_list = list()
        for i in range(len(ann[1])):
            pre_ = np.zeros((1000,1000))
            pre_[pre[:,:,0]>0] = 1#predictで0以上のindexだけ1にする
            pre_ = np.reshape(pre_,[-1])
            X = np.zeros((1000,1000))
            X[ann[1][i][1]] = 1#真値をマスクにする
            cm = confusion_matrix(np.reshape(X,[-1]),pre_)

            TP = cm[0][0]
            FP = cm[0][1]
            FN = cm[1][0]
            TP = cm[1][1]
            iou = TP/(TP+FP+FN)
            iou_list.append(iou)
        max_index = np.argmax(np.array(iou_list))#最もiouが高いindexの取得
        X = np.zeros((1000,1000))
        X[ann[1][max_index][1]] = 255
        X = np.tile(np.transpose(X[np.newaxis],[1,2,0]),[1,1,3])
        #pdb.set_trace()
        return X,iou_list[max_index],max_index


def make_result(out,this_batch,threshold):

    batch_size = len(this_batch)

    with open('data/ann_anchor_data/mask_anchor_k.pickle', 'rb') as f:
        anchor = pickle.load(f)
    with open('data/ann_anchor_data/max_min_k.pickle','rb') as f:
        X = pickle.load(f)

    with open('data/ann_anchor_data/ann_coords_4_T.pickle','rb') as f:
        annotations = pickle.load(f)


    t_0_max = np.array(X[0])
    t_0_min = np.array(X[1])
    t_1_max = np.array(X[2])
    t_1_min = np.array(X[3])

    iou_return = list()
    precision_return = list()
    recall_return = list()
    iou_label =  [[] for i in range(6)]
    precision_label = [[] for i in range(6)]
    recall_label = [[] for i in range(6)]
    T_0_list = list()
    T_1_list = list()
    R_list = list()
    max_index = list()
    idx = list()
    for i in range(batch_size):
        #pdb.set_trace()
        out_now = np.transpose(np.reshape(out[i],[361,5,6]),[2,0,1]) #networkの出力をreshape
        #pdb.set_trace()
        image_name = this_batch[i]#画像の名前

        out_conf = np.reshape(out_now[3],[-1])#confidenceの抽出

        confidence = (1/(1+np.exp(-out_conf)))#confidenceを0~1で表現
        #pdb.set_trace()
        trast_conf = np.where(confidence>threshold)[0]#一定数以上のものを予測とすし、trast_confとする
        
        print('size trast conf:{0}'.format(len(trast_conf)))
        print('trast conf:{0}'.format(trast_conf))
        #pdb.set_trace()
        pre = np.zeros((1000,1000))
        if len(trast_conf) == 0:
            continue
        predict = list()
        true = list()
        true_list = list()
        trast_conf_new = list()
        
        for ann in annotations:
            if ann[0] == image_name:
                #annotationsから入力された画像と同じ名前のものを選択
                ann_num = ann
                break
        max_index.extend(trast_conf//361)
        idx.extend(trast_conf%361)

        """
        for j in range(len(trast_conf)):
            trast_conf_new.append(trast_conf[j])
            if j == 0:
                mask_anchor_all = np.reshape(anchor,[1805,1000,1000])[trast_conf[j]][np.newaxis]
            else:
                mask_anchor_all = np.append(mask_anchor_all,np.reshape(anchor,[1805,1000,1000])[trast_conf[j]][np.newaxis],axis=0)
        ma = mask_anchor_all
        for j in range(len(trast_conf)):
            anchor_now = np.tile(np.reshape(anchor,[1805,1000,1000])[trast_conf[j]][np.newaxis],[len(trast_conf),1,1])

            or_ = np.logical_or(anchor_now,mask_anchor_all).astype(np.int)
            and_ = np.logical_and(anchor_now,mask_anchor_all).astype(np.int)
            or__ = np.sum(np.sum(or_,1),1)
            and_ = np.sum(np.sum(and_,1),1)
            iou = and_/or__
            for x in range(len(iou)):
                if iou[x] == 1:
                    continue
                elif iou[x] > 0.3:
                    ma = np.append(ma,or_[x][np.newaxis],axis=0)
                    trast_conf_new.append(trast_conf[x])
                    #print(x)
            if j == 5:
                break
        ma = ma.tolist()
        mask_anchor_all = list()
        trast_conf = list()
        for j in range(len(ma)):#同じ真値が選択されていた場合true_listの重複を消去
            if mask_anchor_all.count(ma[j])>0:
                pass
            else:
                mask_anchor_all.append(ma[j])
                trast_conf.append(trast_conf_new[j])
        """
        """
        #pdb.set_trace()
        for j in range(len(trast_conf)):
            #print(confidence[trast_conf[j]])
            R = np.reshape(out_now[0],[-1])[trast_conf[j]]

            T_0 = np.dot(np.divide(np.reshape(out_now[1],[-1])[trast_conf[j]]+1,2),t_0_max-t_0_min)+t_0_min #T_0の正規化をもとの値に戻す
            T_1 = np.dot(np.divide(np.reshape(out_now[2],[-1])[trast_conf[j]]+1,2),t_1_max-t_1_min)+t_1_min #T_1の正規化をもとの値に戻す

            anchor_now = np.reshape(anchor,[1805,1000,1000])[trast_conf[j]]#trast_confのindexにあるanchorを取得
            
            #affine = np.array([[1,0,-T_1],[0,1,-T_0]])#並進

            #pre=cv2.warpAffine(anchor_now, affine, (1000,1000))

            #affine = cv2.getRotationMatrix2D((0,0),-R,1.0)#回転

            #pre=cv2.warpAffine(anchor_now, affine, (1000,1000))
            #pdb.set_trace()
            affine = cv2.getRotationMatrix2D((0,0),math.degrees(R),1.0)
            affine[0][2] += T_0
            affine[1][2] += T_1
            print('T0:{0}   T1:{1}'.format(T_0,T_1))
            pre = cv2.warpAffine(anchor_now,affine,(1000,1000))
            #pre = anchor_now
            #pre = anchor_now
            where_ = np.where(pre)
            pre_1 = np.zeros((1000,1000))
            pre_2 = np.zeros((1000,1000))
            pre_1[where_] = 0
            pre_2[where_] = 200
            pre = pre[np.newaxis]
            pre_1 = pre_1[np.newaxis]
            pre_2 = pre_2[np.newaxis]
            pre = np.append(pre,pre_1,0)
            pre = np.append(pre,pre_2,0)

            pre = np.transpose(pre,[1,2,0])

            #x_ = mean(where_[0])
            #y_ = mean(where_[1])
            #label = where_pre_label(x_,y_,where_)
            ann,iou_parts,delete_index = detect_most_near(pre,ann_num)#ann:予測に最も近い真値 iou_parts:最も大きいiou delete_index:選択された真値
            predict.append(1)#予測配列に1をappend
            #print('iou:{0}'.format(iou_parts))
            if iou_parts == 1000:
                true.append(0)#画像上に物体が存在しないため、予測失敗
                iou_parts = 0
            elif iou_parts > 0.3:
                true.append(1)#iouが閾値より大きいため、予測成功
                true_list.append(delete_index)#選択された真値は、あとでdelete
            else:
                true.append(0)#iouが閾値より小さいため、予測失敗

            T_0_list.append(T_0)
            T_1_list.append(T_1)
            R_list.append(R)

            iou_return.append(iou_parts)
            #iou_label[label-1].append(iou_parts)

            imgcv = cv2.imread(os.path.join('data/VOC2012/sphere_test',this_batch[i]))
            prediction = cv2.addWeighted(np.asarray(imgcv,np.float64),0.7,np.asarray(pre,np.float64),0.3,0)
            prediction = cv2.addWeighted(np.asarray(prediction,np.float64),0.6,np.asarray(ann,np.float64),0.4,0)
            cv2.imwrite('../GoogleDrive/sphereLite/test_image_{0}_{1}.png'.format(this_batch[i][:6],j),prediction)
            
        #pdb.set_trace()
        count_list = list()
        for i in true_list:#同じ真値が選択されていた場合true_listの重複を消去
            if count_list.count(i)>0:
                pass
            else:
                count_list.append(i)
        #pdb.set_trace()
        count_list.sort(reverse=True)

        for i in count_list:#ann_numから既に選択された真値を消去
                del ann_num[1][i]

        [predict.append(0) for i in range(len(ann_num[1]))]#残った真値の数だけ、predictに0をappend
        [true.append(1) for i in range(len(ann_num[1]))]#残った真値の数だけ、trueに1をappend
        precision = precision_score(np.array(true),np.array(predict))#precisionの計算
        #pdb.set_trace()
        if np.sum(np.array(true)) == 0:
            recall = 1
        else:
            recall = recall_score(np.array(true),np.array(predict))#recallの計算
        precision_return.append(precision)
        recall_return.append(recall)
        #pdb.set_trace()
        """
    return iou_return,precision_return,recall_return,T_0_list,T_1_list,R_list,max_index,idx


import math

def where_pre_label(x,y,where_pre):
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


    #threshold = [0.3,0.4,0.5,0.6,0.7]
    threshold = [0.2]
    iou_label_ = ['iou_label_conf_05']
    precision_label = ['precision_conf_05']
    recall_label = ['recall_conf_05']
    #iou_label_ = ['iou_label_conf_03','iou_label_conf_04','iou_label_conf_05','iou_label_conf_06','iou_label_conf_07','iou_label_conf_07']
    #precision_label = ['precision_conf_03','precision_conf_04','precision_conf_05','precision_conf_06','precision_conf_07','precision_conf_07']
    #recall_label = ['recall_conf_03','recall_conf_04','recall_conf_05','recall_conf_06','recall_conf_07','recall_conf_07']

    for k in range(len(threshold)):

        iou_ = list()
        precision_ = list()
        recall_ = list()
        T_0_ = list()
        T_1_ = list()
        R_ = list()
        max_ = list()
        id_ = list()
        iou_label_all =  [[] for i in range(6)]
        precision_label_all = [[] for i in range(6)]
        recall_label_all = [[] for i in range(6)]
        X=0
        print(threshold[k])
        #pdb.set_trace()
        the = threshold[k]
        for j in range(n_batch):
            from_idx = j * batch
            to_idx = min(from_idx + batch, len(all_inps))
            #print(X)
            X = X+1
            # collect images input in the batch
            this_batch = all_inps[from_idx:to_idx]
            inp_feed = pool.map(lambda inp: (
                np.expand_dims(self.framework.preprocess(
                    os.path.join(inp_path, inp)), 0)), this_batch)

            # Feed to the net
            feed_dict = {self.inp : np.concatenate(inp_feed, 0)}
            self.say('Forwarding {0} inputs ...threshold:{1}'.format(len(inp_feed),the))
            start = time.time()
            out = self.sess.run(self.out, feed_dict)

            stop = time.time(); last = stop - start
            self.say('Total time = {}s / {} inps = {} ips'.format(
                last, len(inp_feed), len(inp_feed) / last))

            # Post processing
            self.say('Post processing {} inputs ...'.format(len(inp_feed)))
            start = time.time()

            iou,precision,recall,T_0,T_1,R,max__,id__ = make_result(out,this_batch,the)
            iou_.extend(iou)
            precision_.extend(precision)
            recall_.extend(recall)
            T_0_.extend(T_0)
            T_1_.extend(T_1)
            R_.extend(R)
            max_.extend(max__)
            id_.extend(id__)
            #[iou_label_all[i].extend(iou_label[i]) for i in range(6)]
            if X == 100:
                break
        iou_index = 'data/out_data/'+iou_label_[k]+'.pickle'
        precision_index = 'data/out_data/'+precision_label[k]+'.pickle'
        recall_index = 'data/out_data'+recall_label[k]+'.pickle'
        #pdb.set_trace()
        plt.hist(max_)
        plt.savefig('../GoogleDrive/max_.png')
        plt.clf()
        plt.hist(id_,range=(0,350))
        plt.savefig('../GoogleDrive/id_.png')
        plt.clf()
        pdb.set_trace()

        with open(iou_index,mode = 'wb') as f:
                pickle.dump(iou_label_all,f) 
        with open(precision_index,mode='wb') as f:
                pickle.dump(precision_,f)
        with open(recall_index,mode='wb') as f:
                pickle.dump(recall_,f)
        print("finish threshold:{0}".format(threshold[k]))
        print("iou:{0}".format(np.nanmean(iou_)))
        print("precision:{0}".format(np.nanmean(precision_)))
        print("recall:{0}".format(np.nanmean(recall_)))
        #pdb.set_trace()
        
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
