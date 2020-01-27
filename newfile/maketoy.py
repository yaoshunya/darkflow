import os
import sys
import numpy as np
import pdb
import cv2
import math
import random

if __name == '__main__':
    for i in range(5000):
        mask = np.zeros((1000,1000))
        x_0 = random.randint(50, 250)
        x_1 = random.randint(50, 250)
        y_0 = random.randint(50, 250)
        y_1 = random.randint(50, 250)
        if x_0 < x_1:
            x_min = x_0
            x_max = x_1
        if y_0 < y_1:
            y_min = y_0
            y_max = y_1
        mask[y_min:y_max,x_min,x_max] = 255


