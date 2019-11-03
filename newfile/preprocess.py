import os
import sys
import glob
import numpy as np
import pdb
import cv2
import pickle
import xml.etree.ElementTree as ET

def get_projection_grid(b, grid_type="Driscoll-Healy"):
    theta, phi = np.meshgrid(np.arange(2 * b) * np.pi / (2. * b),np.arange(2 * b) * np.pi / b, indexing='ij')
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_


def rand_rotation_matrix(deflection=1.0,randnums=None):
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta,phi,z=(np.pi/2,np.pi,1)

    r_=np.sqrt(z)
    V=(
    np.sin(phi)*r_,
    np.cos(phi)*r_,
    np.sqrt(2.0-z)
    )

    st=np.sin(theta)
    ct=np.cos(theta)

    R=np.array(((ct,st,0),(-st,ct,0),(0,0,1)))

    M=(np.outer(V,V)-np.eye(3)).dot(R)

    return M

def rotate_grid(rot,grid):
    x,y,z=grid
    xyz=np.array((x,y,z))
    x_r,y_r,z_r=np.einsum('ij,jab->iab',rot,xyz)
    return x_r,y_r,z_r

def sample_bilinear(signal, rx, ry):
    #pdb.set_trace()
    if len(signal.shape) > 2:
        signal_dim_x = signal.shape[1]
        signal_dim_y = signal.shape[2]
    else:
        signal_dim_x = signal.shape[0]
        signal_dim_y = signal.shape[1]
    rx *= signal_dim_x
    ry *= signal_dim_y


    ix = rx.astype(int)
    iy = ry.astype(int)

    ix0 = ix - 1
    iy0 = iy - 1
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    return (iy1 - ry) * fx1 + (ry - iy0) * fx2

def project_2d_on_sphere(signal, grid , projection_origin=None):


    if projection_origin is None:
        projection_origin = (0, 0, 2 + 1e-3)

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)

    sample = sample_bilinear(signal, rx, ry)


    sample *= (grid[2] <= 1).astype(np.float64)

    if len(sample.shape) > 2:
        sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
        sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)
        sample = (sample - sample_min) / (sample_max - sample_min)
    else:
        sample_min = sample.min(axis=(0,1))
        sample_max = sample.max(axis=(0,1))
        sample = (sample - sample_min) / (sample_max - sample_min)

    sample *= 255
    sample = sample.astype(np.uint8)

    return sample

def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds
    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample

def project_sphere_on_xy_plane(grid, projection_origin):
    sx, sy, sz = projection_origin
    x, y, z = grid

    z = z.copy() + 1


    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))

    return rx, ry

def mask_anchor(anchor,H):
    img_x = 1000
    img_y = 1000

    step_x=0
    step_y=0
    S = H
    #anchor = np.reshape(anchor,(5,2))
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

                w_ = float(x[l].split(",")[0])
                h_ = float(x[l].split(",")[1])

                mask_base = np.zeros((img_x,img_y),dtype=int)

                #side = int(w_*1224/488)
                #ver = int(h_*370/488)
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

                #-------------------------------------------------
                #アンカーの球面写像
                grid = get_projection_grid(b=500)
                rot = rand_rotation_matrix(deflection=1.0)
                grid = rotate_grid(rot,grid)
                mask_base = project_2d_on_sphere(mask_base,grid)
                #-------------------------------------------------

                #resize_mask = np.resize(mask_base,(H,H))
                resize_mask = mask_base
                if l == 0:
                    mask = resize_mask[np.newaxis]
                else:
                    mask = np.append(mask,resize_mask[np.newaxis],axis=0)


            step_x += step_size
            if t == 0:
                mask_ = mask[np.newaxis]
            else:
                mask_ = np.append(mask_,mask[np.newaxis],axis=0)
        print(i)
        step_y += step_size
        if i == 0:
            mask_fi = mask_[np.newaxis]
            #mask_fi_row = mask__row[np.newaxis]
        else:
            mask_fi = np.append(mask_fi,mask_[np.newaxis],axis=0)
            #mask_fi_row = np.append(mask_fi_row,mask__row[np.newaxis],axis=0)
    """
    with open('data/anchor/anchor.binaryfile','wb') as anc:
        pickle.dump(mask_fi_row,anc,protocol=4)
    """
    return mask_fi

def pascal_voc_clean_xml(ANN, pick, exclusive = False):
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')
    size = len(annotations)
    mask = np.array([])

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i+1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        # actual parsing
        in_file = open(file)
        tree=ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text)
        t = 0

        all = list()

        for obj in root.iter('object'):
            current = list()
            name = obj.find('name').text
            if name not in pick:
                continue
            
            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))

            #make mask annotation from coordinates
            ###############################################################
            mask_prepare = np.zeros((375,1242),dtype=int)
            mask_prepare[yn:yx,xn:xx]=255
            mask_parts = np.array([])
            grid = get_projection_grid(b=500)
            rot = rand_rotation_matrix(deflection=1.0)
            grid = rotate_grid(rot,grid)
            mask_parts = project_2d_on_sphere(mask_prepare,grid)
            #mask_parts = cv2.resize(mask_parts,(19,19))
            mask_parts = np.reshape(mask_parts,[-1])
            #pdb.set_trace()
            ################################################################
            ################################################################
            #pdb.set_trace()
            #mask_parts = np.reshape(mask_parts,[1000,1000])
            
            mask_ = mask_parts[np.newaxis]
            
            current = [name,mask_]
            all += [current]
    
        add = [[jpg,all]]
        #pdb.set_trace()
        dumps += add
        in_file.close()
        #pdb.set_trace()
    os.chdir(cur_dir)
    all = list()
    current = list()
    add = list()
    return np.array(dumps)

def make_coords_from_mask(data,flag):

    if flag == 0:
        anchor = data
        anchor = np.reshape(anchor,[anchor.shape[0]*anchor.shape[1],anchor.shape[2],anchor.shape[3],anchor.shape[4]])
        anchor_coords = list()
        anc_parts = []
        
        for i in range(anchor.shape[0]):
            for j in range(anchor.shape[1]):
               anchor_coords_parts = np.array(np.where(anchor[i][j]>0))
               anc_parts.append(anchor_coords_parts)
            anchor_coords.append(anc_parts)
        return anchor_coords

    if flag == 1:
        ann_coords = []
        annotations = data
        
        #pdb.set_trace()
        for i in range(annotations.shape[0]):      
            #pdb.set_trace()
            jpg = annotations[i][0]
            all = []
            for j in range(len(annotations[i][1])):
                #pdb.set_trace()
                #print(j)
                name = annotations[i][1][j][0]
                current = list()
                ann_coords_parts = np.where(np.reshape(annotations[i][1][j][1],[1000,1000])>0)
                current = [name,ann_coords_parts]
                all += [current]

            add = [[jpg,all]]
            ann_coords += add
        
        #pdb.set_trace()

        return ann_coords
    return 0

if __name__ ==  '__main__':

    #----------------------------------------
    #マスクアンカーが存在しなければ作成し、pickleファイルで保存
    #ファイルがあれば読み込み
    if not os.path.exists('mask_anchor.pickle'):
        print("make mask anchor")
        with open("anchor.txt") as f:
            x = f.read().split()

        anchor = mask_anchor(np.array(x),19)

        with open('mask_anchor.pickle',mode = 'wb') as f:
            pickle.dump(anchor,f)
        print("finish making mask anchor")
    else:
        """
        with open("mask_anchor.pickle",mode = 'rb') as f:
            anchor = pickle.load(f)
        """
        pass
        #print("load mask anchor")
        #with open('mask_anchor.pickle',mode = 'rb') as f:
        #    anchor = pickle.load(f)
    #----------------------------------------

    #----------------------------------------
    #マスクアノテーションが存在しなければ作成し、pickleファイルで保存
    #ファイルがあれば読み込み
    #
    if not os.path.exists('ann_coords_1.pickle'):
        print("make mask annotations_1")
        path = '../data/VOC2012/AnnotationsTrain_1' #残差を計算したい対象
        pick = ['car','Truck'] #見つけたい物体

        annotations = pascal_voc_clean_xml(path,pick)
        #annotations = np.array(annotations).astype('int')

        print("make mask annotations coords_1")
	
        ann_coords=make_coords_from_mask(annotations,1)

        with open('ann_coords_1.pickle',mode = 'wb') as f:
            pickle.dump(ann_coords,f)
        print("finish making mask annotations coords")

        del annotations,ann_coords

    #elif not os.path.exists('mask_annotations_2.pickle'):
        print("make mask annotations_2")
        path = '../data/VOC2012/AnnotationsTrain_2' #残差を計算したい対象
        pick = ['car','Truck'] #見つけたい物体
        annotations = pascal_voc_clean_xml(path,pick)

        ann_coords=make_coords_from_mask(annotations,1)

        with open('ann_coords_2.pickle',mode = 'wb') as f:
            pickle.dump(ann_coords,f)
        print("finish making mask annotations coords")
        
        del annotations,ann_coords

    #elif not os.path.exists('mask_annotations_3.pickle'):
        print("make mask annotations_3")
        path = '../data/VOC2012/AnnotationsTrain_3/AnnotationsTrain_1' #残差を計算したい対象
        pick = ['car','Truck'] #見つけたい物体
        annotations = pascal_voc_clean_xml(path,pick)

        ann_coords=make_coords_from_mask(annotations,1)

        with open('ann_coords_3.pickle',mode = 'wb') as f:
            pickle.dump(ann_coords,f)
        print("finish making mask annotations coords")

        del annotations,ann_coords

    #elif not os.path.exists('mask_annotations_4.pickle'):
        print("make mask annotations_4")
        path = '../data/VOC2012/AnnotationsTrain_4/AnnotationsTrain_2' #残差を計算したい対象
        pick = ['car','Truck'] #見つけたい物体
        annotations = pascal_voc_clean_xml(path,pick)

        ann_coords=make_coords_from_mask(annotations,1)

        with open('ann_coords_4.pickle',mode = 'wb') as f:
            pickle.dump(ann_coords,f)
        print("finish making mask annotations coords")

        del annotations,ann_coords

        with open("mask_anchor.pickle",mode = 'rb') as f:
            anchor = pickle.load(f)
        
        anchor_coords=make_coords_from_mask(anchor,0)
        
        with open('anchor_coords.pickle',mode = 'wb') as f:
            pickle.dump(anchor_coords,f)
        
    else:
        with open('anchor_coords.pickle',mode = 'rb') as f:
            anchor = pickle.load(f)
        with open('ann_coords.pickle',mode = 'rb') as f:
            ann = pickle.load(f)
        pdb.set_trace()
        #del annotations_1,annotations_2
        """
        with open('mask_annotations.pickle',mode = 'wb') as f:
            pickle.dump(annotations,f)
        print("finish making mask annotations")
        """



