"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob
#import lie_learn.spaces.S2 as S2
import numpy as np
import pdb

def _pp(l): # pretty printing
    for i in l: print('{}: {}'.format(i,l[i]))

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

def pascal_voc_clean_xml(ANN, pick, exclusive = False):
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')
    size = len(annotations)

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
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
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
                #pdb.set_trace()
                mask_parts = project_2d_on_sphere(mask_prepare,grid)
                ###############################################################
                ###############################################################
                current = [name,mask_parts]
                #pdb.set_trace()
                all += [current]

        add = [[jpg, [w, h, all]]]
        dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]]+=1
                else:
                    stat[current[0]] =1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps
