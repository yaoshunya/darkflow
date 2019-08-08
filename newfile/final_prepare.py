from PIL import Image
import os
import cv2
import numpy as np
import re
import pdb
import glob
import pickle
import argparse
#import lie_learn.spaces.S2 as S2
#from torchvision import datasets
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

"""
os.chdir('../data/train/images')
dir = os.getcwd()
files = os.listdir(dir)

len_images = len(files) #フォルダ内の画像の数
os.chdir('../../../code')
"""




def rand_rotation_matrix(deflection=1.0,randnums=None):
	if randnums is None:
		randnums = np.random.uniform(size=(3,))

	theta,phi,z=(np.pi/2,np.pi,1)
	#theta,phi,z=(np.pi,np.pi/2,1)

	#theta,phi,z=(np.pi,2*np.pi/2.0,0.9)

	r_=np.sqrt(z)
	V=(
		np.sin(phi)*r_,
		np.cos(phi)*r_,
		np.sqrt(2.0-z)
	)
	#pdb.set_trace()

	st=np.sin(theta)
	ct=np.cos(theta)

	R=np.array(((ct,st,0),(-st,ct,0),(0,0,1)))

	M=(np.outer(V,V)-np.eye(3)).dot(R)

	#pdb.set_trace()
	#with open('sample.txt',mode='w') as f:
	#	f.write(str(theta))
	#	f.write(str(phi))
	#	f.write(str(z))
	#	f.write('\n')

	#pdb.set_trace()

	return M

def rotate_grid(rot,grid):
	x,y,z=grid
	xyz=np.array((x,y,z))
	x_r,y_r,z_r=np.einsum('ij,jab->iab',rot,xyz)
	return x_r,y_r,z_r

def get_projection_grid(b, grid_type="Driscoll-Healy"):
	theta, phi = meshgrid(b=b, grid_type=grid_type)
	x_ = np.sin(theta) * np.cos(phi)
	y_ = np.sin(theta) * np.sin(phi)
	z_ = np.cos(theta)
	#pdb.set_trace()
	return x_, y_, z_

def project_sphere_on_xy_plane(grid, projection_origin):
	sx, sy, sz = projection_origin
	x, y, z = grid
	#x = np.append(x[np.newaxis],y[np.newaxis],axis=0)
	#x = np.append(x,z[np.newaxis],axis=0)
	#cv2.imshow('sample',x.T)
	#cv2.waitKey(0)
	#if(np.all(z < 0)):
	#	print("z<0")
	z = z.copy() + 1
	#x = x.copy() + 1
	#if(np.all(z < 0)):
	#	print("z_<0")
	#x = x.copy() + 1#

	t = -z / (z - sz)
	qx = t * (x - sx) + x
	qy = t * (y - sy) + y

	xmin = 1/2 * (-1 - sx) + -1
	ymin = 1/2 * (-1 - sy) + -1

	rx = (qx - xmin) / (2 * np.abs(xmin))
	ry = (qy - ymin) / (2 * np.abs(ymin))
	#rx = qx
	#ry = qy
	#pdb.set_trace()

	return rx, ry

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
	#cv2.imshow('smaple',((iy1 - ry) * fx1 + (ry - iy0) * fx2)[0])
	#cv2.waitKey()

	#fig = plt.figure(figsize=plt.figaspect(1.))
	#ax = fig.add_subplot(111, projection='3d')
	#pdb.set_trace()

	return (iy1 - ry) * fx1 + (ry - iy0) * fx2

def project_2d_on_sphere(signal, grid , projection_origin=None):


	if projection_origin is None:
		projection_origin = (0, 0, 2 + 1e-3)
	#pdb.set_trace()
	#projection=np.array(projection_origin)[:,np.newaxis]
	#plot_sphere(projection)
	rx, ry = project_sphere_on_xy_plane(grid, projection_origin)

	sample = sample_bilinear(signal, rx, ry)
	#pdb.set_trace()

	sample *= (grid[2] <= 1).astype(np.float64)
	#pdb.set_trace()
	if len(sample.shape) > 2:
		sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
		sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)
		sample = (sample - sample_min) / (sample_max - sample_min)
	else:
		sample_min = sample.min(axis=(0,1))
		sample_max = sample.max(axis=(0,1))
		sample = (sample - sample_min) / (sample_max - sample_min)
	#pdb.set_trace()
	sample *= 255
	sample = sample.astype(np.uint8)

	return sample

def divide_color(image):
	image_b = np.array([])
	image_r = np.array([])
	image_g = np.array([])

	for i in range(image.shape[0]):
		if i ==0 :
			image_b = image[i].T[0].T[np.newaxis]
			image_g = image[i].T[1].T[np.newaxis]
			image_r = image[i].T[2].T[np.newaxis]
		else:
			image_b = np.append(image_b,image[i].T[0].T[np.newaxis],axis=0)
			image_g = np.append(image_g,image[i].T[1].T[np.newaxis],axis=0)
			image_r = np.append(image_r,image[i].T[2].T[np.newaxis],axis=0)
		print(i)
	return image_b,image_g,image_r

def create_sphere(data,grid):
	#pdb.set_trace()
	signals = data.reshape(-1,data.shape[1],data.shape[2]).astype(np.float64)
	n_signals = signals.shape[0]
	projections = np.ndarray(
		(signals.shape[0],2*500,2*500),dtype=np.uint8
	)
	current = 0
	#pdb.set_trace()
	while current < n_signals:
		idxs = np.arange(current,min(n_signals,current+5))
		chunk = signals[idxs]
		projections[idxs] = project_2d_on_sphere(chunk,grid)
		current += 5
		print(current)

	return projections

def plot_sphere(grid):
	fig = plt.figure(figsize=plt.figaspect(1.))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.set_aspect('equal')
	ax.plot_surface(grid[0], grid[1], grid[2])
	#ax.scatter(grid[0],grid[1],grid[2])
	#ax.set_axis_off()
	plt.savefig('sphere2.png')
	plt.show()

def meshgrid(b, grid_type='Driscoll-Healy'):

    return np.meshgrid(*linspace(b, grid_type), indexing='ij')


def linspace(b, grid_type='Driscoll-Healy'):
    if grid_type == 'Driscoll-Healy':
        beta = np.arange(2 * b) * np.pi / (2. * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == 'SOFT':
        beta = np.pi * (2 * np.arange(2 * b) + 1) / (4. * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == 'Clenshaw-Curtis':
        # beta = np.arange(2 * b + 1) * np.pi / (2 * b)
        # alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
        # Must use np.linspace to prevent numerical errors that cause beta > pi
        beta = np.linspace(0, np.pi, 2 * b + 1)
        alpha = np.linspace(0, 2 * np.pi, 2 * b + 2, endpoint=False)
    elif grid_type == 'Gauss-Legendre':
        x, _ = leggauss(b + 1)  # TODO: leggauss docs state that this may not be only stable for orders > 100
        beta = np.arccos(x)
        alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
    elif grid_type == 'HEALPix':
        #TODO: implement this here so that we don't need the dependency on healpy / healpix_compat
        from healpix_compat import healpy_sphere_meshgrid
        return healpy_sphere_meshgrid(b)
    elif grid_type == 'equidistribution':
        raise NotImplementedError('Not implemented yet; see Fast evaluation of quadrature formulae on the sphere.')
    else:
        raise ValueError('Unknown grid_type:' + grid_type)
    return beta, alpha



def main():

	os.chdir('../data/VOC2012')
	files = glob.glob("JPEGLite/*")

	#images = []
	images = np.array([])
	image_name = np.array([])
	t=0
	size = (1242,375)
	size_ = (500,500)
	for i in files:
		img = cv2.imread(i)

		if(img.shape[1] != 1242):
			img = cv2.resize(img,size)

		if(t == 0):
			images = np.array(img[np.newaxis])
		else:

			images = np.append(images,img[np.newaxis],axis=0)

		print(i)
		image_name = np.append(image_name,i[8:])

		t += 1
		#print(t)
		if t==10:

			img_b,img_g,img_r = divide_color(images)

			grid = get_projection_grid(b=500)

			rot = rand_rotation_matrix(deflection=1.0)

			rotated_grid = rotate_grid(rot,grid)

			img_b = create_sphere(img_b,rotated_grid)
			img_g = create_sphere(img_g,rotated_grid)
			img_r = create_sphere(img_r,rotated_grid)
			os.chdir('sphere_data')

			for i in range(images.shape[0]):
				#print(i)
				image_sample = np.append(img_b[i][np.newaxis],img_g[i][np.newaxis],axis=0)
				image_sample = np.append(image_sample,img_r[i][np.newaxis],axis=0)
                                #ipdb.set_trace()
				#cv2.imwrite(image_name[i],image_sample.T)
				#pdb.set_trace()
				cv2.imwrite(image_name[i][1:],image_sample.T)
				#cv2.waitKey(0)
			#pdb.set_trace()
			os.chdir('../')
			image_sample = []
			images=np.array([])
			image_name=np.array([])
			t=0


	img_b,img_g,img_r=divide_color(images)
	grid = get_projection_grid(b=500)
	rot = rand_rotation_matrix(deflection=1.0)
	rotated_grid=rotate_grid(rot,grid)

	img_b = create_sphere(img_b,rotated_grid)
	img_g = create_sphere(img_g,rotated_grid)
	img_r = create_sphere(img_r,rotated_grid)


	for i in range(images.shape[0]):
		print(i)
		image_sample = np.append(img_b[i][np.newaxis],img_g[i][np.newaxis],axis=0)
		image_sample = np.append(image_sample,img_r[i][np.newaxis],axis=0)
		cv2.imwrite(os.path.join('sphere_data',image_name[i]),image_sample.T)



if __name__ == '__main__':
	main()
"""
cv2.imshow('image_sample',images[0])
cv2.waitKey(0)
"""
#pdb.set_trace()
