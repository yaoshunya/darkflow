import os
import cv2
import numpy as np
import matplotlib.pylab as plt
import pdb
import glob

def mask_processing(img):
    files = glob.glob("bb/*")

    for i in files:
        mask = cv2.imread(i)
        #pdb.set_trace()
        dst = cv2.bitwise_and(img,mask)
        cv2.imwrite(i,dst)
        cv2.waitKey(0) & 0xFF



def main():
    img = cv2.imread("../../kitti/data/train/000008.png")
    #pdb.set_trace()
    img_x=img.shape[0]  #1000
    img_y=img.shape[1]  #1000

    S = 10 #分割グリッド数

    step_size = int(img_x/S)

    with open("anchor.txt") as f:
        x = f.read().split()
    anchor_size = len(x)

    step_x=0
    step_y=0

    for i in range(S):
        for t in range(S):
            if t==0:
                step_x = 0
            center_x = int((step_x + (step_x + step_size))/2)
            center_y = int((step_y + (step_y + step_size))/2)
            for l in range(anchor_size):

                w_ = float(x[l].split(",")[0])
                h_ = float(x[l].split(",")[1])

                mask_base = np.zeros((img_x,img_y),dtype=int)

                side = int(step_size/w_)
                ver = int(step_size/h_)

                pdb.set_trace()

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
                #mask_image = np.tile(mask_base,(3,1,1)).T
                #pdb.set_trace()
                #mask_base = cv2.resize(mask_base,img.shape[1::-1])[np.newaxis]
                #pdb.set_trace()
                #mask = np.append(np.append(mask_base,mask_base,axis=0),mask_base,axis=0)
                #pdb.set_trace()
                #image = cv2.bitwise_and(img,mask_base)
                cv2.imwrite('bb/grid_{0}{1}_{2}.png'.format(i,t,l),mask_base)
                cv2.waitKey(0) & 0xFF
                #pdb.set_trace()
            step_x += step_size

        step_y += step_size

    mask_processing(img)


if __name__ == '__main__':
	main()
