import numpy as np
import cv2 as cv
import os
from multiprocessing import  Pool

from IntrinsicParametersFunctions.calculate_intrinsic_parameters import calculate_intrinsic_parameters, calculate_seglen

parentFolder = '../cutout_data'
subFolder = ['images', 'points']

imagesFolder = parentFolder + '/' + subFolder[0]
pointsFolder = parentFolder + '/' + subFolder[1]

outputsFolder = '../outputs/'
recreatedOutputsFolder = '../outputs_recreated/'
# preparing the data

imagesList = os.listdir(imagesFolder)
imagesList.sort()
pointsList = os.listdir(pointsFolder)
pointsList.sort()


def calculateAndSaveData(idx):
    imPath = imagesFolder + '/' + imagesList[idx]
    pointsPath = pointsFolder + '/' + pointsList[idx]

    im = cv.imread(imPath)
    points = np.load(pointsPath)    
    
    name = pointsPath.split('/')[-1]
    name = name[:-4] + '.png'
    recreatedIm = calculate_intrinsic_parameters(im, points)
    
    #np.save(outputsFolder + name, x)
    cv.imwrite(recreatedOutputsFolder + name, recreatedIm)
    
poolObj = Pool()
poolObj.map(calculateAndSaveData, range(0, 15))
#poolObj.map(calculateAndSaveData, range(0, len( imagesList)))
poolObj.close()


def calculateSeglen(idx):
    imPath = imagesFolder + '/' + imagesList[idx]
    pointsPath = pointsFolder + '/' + pointsList[idx]

    im = cv.imread(imPath)
    points = np.load(pointsPath)

    seglen = calculate_seglen(im, points)
    return seglen

#seglens = []
#for idx in range(len(pointsList)):
#    seglen = calculateSeglen(idx)
#    seglens.append(seglen)
#
#np.save(outputsFolder + 'seglens.npy', np.array(seglens))

