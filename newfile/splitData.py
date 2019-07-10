import pandas as pd
import numpy as np
import shutil
import os
import pdb

dfTrain = pd.read_csv('../../data/data_voc/VOC2012/ImageSets/Main/trainval.txt',header=None)
dfTest = pd.read_csv('../../data/data_voc/VOC2012/ImageSets/Main/test.txt',header=None)
imgDir = '../../data/data_voc/VOC2012/JPEGImages'
trainImgDir = '../data/PNGImagesTrain'
testImgDir = '../data/PNGImagesTest'
annoDir = '../../data/data_voc/VOC2012/Annotations'
trainAnnoDir = '../data/AnnotationsTrain'
testAnnoDir = '../data/AnnotationsTest'

# Train data
for ind in np.arange(len(dfTrain)):

	#---------------------
	# folder pathes
	# images
	fname = '%06d.png' % (dfTrain.values[ind][0])
	imgPath = os.path.join(imgDir,fname)
	trainImgPath = os.path.join(trainImgDir,fname)

	# annotations
	fname = '%06d.xml' % (dfTrain.values[ind][0])
	annoPath = os.path.join(annoDir,fname)
	trainAnnoPath = os.path.join(trainAnnoDir,fname)
	#---------------------

	#---------------------
	# copy files
	print("{}=>{}".format(imgPath,trainImgPath))
	shutil.copyfile(imgPath,trainImgPath)

	print("{}=>{}".format(annoPath,trainAnnoPath))
	shutil.copyfile(annoPath,trainAnnoPath)
	#---------------------

# Test data
for ind in np.arange(len(dfTest)):

	#---------------------
	# folder pathes
	# images
	fname = '%06d.png' % (dfTest.values[ind][0])
	imgPath = os.path.join(imgDir,fname)
	testImgPath = os.path.join(testImgDir,fname)

	# annotations
	fname = '%06d.xml' % (dfTest.values[ind][0])
	annoPath = os.path.join(annoDir,fname)
	testAnnoPath = os.path.join(testAnnoDir,fname)
	#---------------------

	#---------------------
	# copy files
	print("{}=>{}".format(imgPath,testImgPath))
	shutil.copyfile(imgPath,testImgPath)

	print("{}=>{}".format(annoPath,testAnnoPath))
	shutil.copyfile(annoPath,testAnnoPath)
	#---------------------

