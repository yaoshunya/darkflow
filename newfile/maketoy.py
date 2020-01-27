import os
import sys
import numpy as np
import pdb
import cv2
import math
import random

if __name__ == '__main__':
    for i in range(5000):
        mask = np.zeros((1000,1000))
        center_x = random.randint(0,1000)
        center_y = random.randint(0,1000)
        side = random.randint(0, 250)
        ver = random.randint(0,250)

        side_min = center_x - side
        side_max = center_x + side
        ver_min = center_y - ver
        ver_max = center_y + ver

        if side_min < 0:
            side_min = 0
        if side_max > 1000:
            side_max = 1000
        if ver_min < 0:
            ver_min = 0
        if ver_max > 1000:
            ver_max = 1000
        mask[ver_min:ver_max,side_min:side_max] = 255
        grid = get_projection_grid(b=500)
        rot = rand_rotation_matrix(deflection=1.0)
        grid = rotate_grid(rot,grid)
        mask = project_2d_on_sphere(mask,grid).T
        
        i_ = str(n).zfill(5)
        print(i)
        if i > 2000:
            cv2.imwrite(os.path.join('data/VOC2012/toy_test/{0}.png'.format(i_)))
        else:
            cv2.imwrite(os.path.join('data/VOC2012/toy_train/{0}.png'.format(i_)))



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
