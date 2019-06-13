from PIL import Image
import os
import cv2
import numpy as np
import re
import pdb
import glob
import pickle
import argparse
import lie_learn.spaces.S2 as S2
from torchvision import datasets
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
import final_prepare as fi

NORTHPOLE_EPSILON = 1e-3
parser = argparse.ArgumentParser()
parser.add_argument("--bandwidth",
					help="the bandwidth of the S2 signal",
					type=int,
					default=500,
					required=False)
parser.add_argument("--chunk_size",
					help="size of image chunk with same rotation",
					type=int,
					default=5,
					required=False)
parser.add_argument("--noise",
					help="the rotational noise applied on the sphere",
					type=float,
					default=1.0,
					required=False)
args = parser.parse_args()

def create_sphere(data,grid): #data:list grid:tuple

	mask = []
	for i in range(len(data)):
		mask_parts = np.array([])
		"""
		if len(data[i]) > 2:  #一枚の画像に複数個の物体が映っている場合
			for t in range(len(data[i])): #画像と同じ大きさの配列（0）を作成　list
				mask_spherical = fi.project_2d_on_sphere(data[i][t],grid)
				if t == 0:
					mask_parts = mask_spherical
				else:
					mask_parts = np.append(mask_parts,mask_spherical,axis=0)
		else:
			mask_parts = fi.project_2d_on_sphere(data[i][0],grid)
		"""
		#pdb.set_trace()
		mask_parts = fi.project_2d_on_sphere(data[i],grid)
		print(i)
		mask.append(mask_parts)
	return mask


def prepare_mask():
	return np.zeros((375,1242),dtype=int)

def make_mask(label_all):
	mask_all=[]
	for i in range(len(label_all)):
		#mask_parts = np.empty([375,1242])
		mask_parts = np.array([])
		if len(label_all[i]) > 2: #一枚の画像に複数個の物体が映っている場合
			for t in range(len(label_all[i])):
				mask_prepare = prepare_mask() #画像と同じ大きさの配列（0）を作成　list
				minx=int(float(label_all[i][t][4]))
				miny=int(float(label_all[i][t][5]))
				maxx=int(float(label_all[i][t][6]))
				maxy=int(float(label_all[i][t][7]))
				mask_prepare[miny:maxy,minx:maxx]=255
				#print(t)
				#pdb.set_trace()
				if t == 0:
					mask_parts = mask_prepare[np.newaxis]
				else:
					mask_parts = np.append(mask_parts,mask_prepare[np.newaxis],axis=0)
		else:
			mask_prepare = prepare_mask()
			minx=int(float(label_all[i][0][4]))
			miny=int(float(label_all[i][0][5]))
			maxx=int(float(label_all[i][0][6]))
			maxy=int(float(label_all[i][0][7]))
			mask_prepare[miny:maxy,minx:maxx]=255
			#cv2.imshow("sample",mask_prepare)
			#cv2.imwrite('{0}.png'.format(file_num),mask_prepare)
			#cv2.waitKey(0) & 0xFF
			#cv2.waitKey(0)
			#pdb.set_trace()
			mask_parts = mask_prepare[np.newaxis]
		mask_all.append(mask_parts)
	return mask_all

def mask_test():
	files = glob.glob("out/*")
	for i in file:
		file_type = i[:6]
	return 0

def main():

	files = glob.glob("../../kitti/data/label_2/*")
	s_all = []
	file_name = np.array([])
	count = 0
	for i in files:
		s_split=np.array([])
		file_name = np.append(file_name,i[:])
		with open(i) as f:
			s=f.read().splitlines()
			s_numpy=np.array(s)
			if len(s) > 2: #一枚の画像に複数個の物体が映っている場合
				for t in range(len(s)):
					if t == 0:
						s_split = np.array(np.array(s)[t].split())[np.newaxis]
					else:
						s_split = np.append(s_split,np.array(np.array(s)[t].split())[np.newaxis],axis=0)
			else:
				#一枚の画像に物体が一つ映っている場合
				s_split = np.array(s_numpy[0].split())[np.newaxis]
			s_all.append(s_split)
		count += 1
		if count == 50:
			#pdb.set_trace()
			break;

	mask_list = make_mask(s_all) #作成されたマスク配列（numpy）をlistにしたものが返ってくる
	#pdb.set_trace()
	grid = fi.get_projection_grid(b=args.bandwidth)
	rot = fi.rand_rotation_matrix(deflection=args.noise)
	rotated_grid = fi.rotate_grid(rot,grid)

	mask = create_sphere(mask_list,rotated_grid)
	#pdb.set_trace()
	for i in range(len(mask)):
		if len(mask[i]) > 1:
			for t in range(len(mask[i])):
				cv2.imwrite('out/{}_{}.png'.format(str(i).zfill(6),str(t).zfill(2)),mask[i][t].T)
				cv2.waitKey(0) & 0xFF
		else:
			cv2.imwrite('out/{}_00.png'.format(str(i).zfill(6)),mask[i][0].T)
			cv2.waitKey(0) & 0xFF
	#pdb.set_trace()

if __name__ == '__main__':
	main()
