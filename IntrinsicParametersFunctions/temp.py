import numpy as np
import os
from IntrinsicParametersFunctions.f_x_to_model_intrinsic import f_x_to_model_intrinsic
import cv2 as cv

x = [25, 25, 3.14/2, .848, 1.8, 1.08, 1.35, 235, 235 * .83, .64 * .83 * 235, .506, .5144, .4399, .8692, .3155, .7137, .5127, .58, 1.006]
x = np.array(x)
seglen = 3.906
randomize = 0
imageSizeX, imageSizeY = 50, 50
im, _ = f_x_to_model_intrinsic(x, seglen, randomize, imageSizeX, imageSizeY)
cv.imwrite('temp.png', im)




