import numpy as np
from IntrinsicParametersFunctions.draw_anterior_b_intrinsic import draw_anterior_b_intrinsic
from IntrinsicParametersFunctions.gen_lut_b_tail import gen_lut_b_tail
from numpy.random import normal as normrnd
import cv2 as cv

def imGaussNoise(image, mean, var):
    """
       Function used to make image have static noise

       Args:
           image (numpy array): image
           mean (float): mean
           var (numpy array): var

       Returns:
            noisy (numpy array): image with noise applied
       """
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


# imblank = np.zeros((43,38))
# imblank[0,0 ] = 255
# imblank = imblank / 255
# gN1 = (np.random.rand() * np.random.normal(50, 10)) / 255
# gN2 = (np.random.rand() * 50 + 20) / 255 ** 2
# imblank = imGaussNoise(imblank, gN1, gN2)
# imblank *= 255
# imblank[0,0] = 0
# imblank_noise = imblank.copy()

def f_x_to_model_intrinsic(x, seglen, randomize, imageSizeX, imageSizeY):

    hp = x[0: 2]
    dt = np.array([x[2], 0, 0, 0, 0, 0, 0, 0, 0])
    pt = np.zeros((2, 10))
    theta = np.zeros((9,1))
    theta[0] = dt[0]
    pt[:,0] = hp

    for n in range(0, 9):
        R = np.array([[np.cos(dt[n]), -np.sin(dt[n])], [np.sin(dt[n]), np.cos(dt[n])]])
        if n == 0:
            vec = np.matmul(R, np.array([seglen, 0], dtype=R.dtype))
        else:
            vec = np.matmul(R, vec)
            theta[n] = theta[n - 1] + dt[n]
        pt[:, n + 1] = pt[:, n] + vec

    # TODO: should I wrap
    theta = theta % ( 2 * np.pi)

    size_lut = 49
    size_half = (size_lut + 1) / 2
    dh1 = pt[0, 1] - np.floor(pt[0, 1])
    dh2 = pt[1, 1] - np.floor(pt[1, 1])

    fish_anterior, _, _, _, _, _, _  = (draw_anterior_b_intrinsic(seglen, x[2], 0, 0, dh1, dh2, 2, size_lut, randomize, x[3:16]))
    imblank = np.zeros((int(imageSizeY), int(imageSizeX)), dtype=np.uint8)
    # imblankcopy = imblank.copy()
    headpix = imblank.copy()
    bodypix = imblank.copy()
    coor_h1 = np.floor(pt[0, 1])
    coor_h2 = np.floor(pt[1, 1])
    # headpix[int(np.maximum(0, coor_h2 - (size_half - 1))): int(np.minimum((imageSizeY), 1 + coor_h2 + (size_half - 1))),
    # int(np.maximum(0, coor_h1 - (size_half - 1))): int(
    #     np.minimum((imageSizeX), 1 + coor_h1 + (size_half - 1)))] = fish_anterior[
    #                                                                 int(np.maximum((size_half - 1) - coor_h2, 0)): int(
    #                                                                     np.minimum(
    #                                                                         (imageSizeY) - coor_h2 + size_half - 1,
    #                                                                         size_lut)),
    #                                                                 int(np.maximum((size_half - 1) - coor_h1, 0)): int(
    #                                                                     np.minimum(
    #                                                                         (imageSizeX) - coor_h1 + size_half - 1,
    #                                                                         size_lut))]

    headpix[int(np.maximum(1, coor_h2 - (size_half - 1))) -1: int(np.minimum((imageSizeY), coor_h2 + (size_half - 1))),
    int(np.maximum(1, coor_h1 - (size_half - 1))) -1: int(np.minimum((imageSizeX), coor_h1 + (size_half - 1)))] = \
        fish_anterior[ int(np.maximum((size_half + 1) - coor_h2, 1)) -1: int( np.minimum((imageSizeY) - coor_h2 + size_half , size_lut)),
                        int(np.maximum((size_half + 1) - coor_h1, 1)) - 1: int( np.minimum( (imageSizeX) - coor_h1 + size_half, size_lut))]


    size_lut = 29
    size_half = (size_lut + 1) / 2

    for ni in range(0, 7):
        n = ni + 2
        coor_t1 = np.floor(pt[0, n])
        dt1 = pt[0, n] - coor_t1
        coor_t2 = np.floor(pt[1, n])
        dt2 = pt[1, n] - coor_t2
        # tailpix = imblankcopy
        tailpix = imblank
        tail_model = gen_lut_b_tail(ni, seglen, dt1, dt2, theta[n], randomize,x[16:19])
        # tailpix[
        # int(np.maximum(0, coor_t2 - (size_half - 1))): int(np.minimum(imageSizeY, 1 + coor_t2 + (size_half - 1))),
        # int(np.maximum(0, coor_t1 - (size_half - 1))): int(
        #     np.minimum(imageSizeX, 1 + coor_t1 + (size_half - 1)))] = tail_model[
        #                                                               int(np.maximum((size_half - 1) - coor_t2,
        #                                                                              0)): int(np.minimum(
        #                                                                   imageSizeY - coor_t2 + size_half - 1,
        #                                                                   size_lut)),
        #                                                               int(np.maximum((size_half - 1) - coor_t1,
        #                                                                              0)): int(np.minimum(
        #                                                                   imageSizeX - coor_t1 + size_half - 1,
        #                                                                   size_lut))]

        tailpix[ int(np.maximum(1, coor_t2 - (size_half - 1))) -1: int(np.minimum(imageSizeY, coor_t2 + (size_half - 1))),
        int(np.maximum(1, coor_t1 - (size_half - 1))) -1: int( np.minimum(imageSizeX, coor_t1 + (size_half - 1)))] = \
            tail_model[ int(np.maximum((size_half + 1) - coor_t2,1)) -1: int(np.minimum(imageSizeY - coor_t2 + size_half ,size_lut)),
                        int(np.maximum((size_half + 1) - coor_t1, 1)) -1: int(np.minimum(imageSizeX - coor_t1 + size_half ,size_lut))]


        bodypix = np.maximum(bodypix, tailpix)

    graymodel = np.maximum(headpix, (randomize * normrnd(1, 0.1) + (1 - randomize)  ) * bodypix)

    # add_noise = True
    # if add_noise:
    # graymodel = graymodel/255
    # gN1 = (np.random.rand() * np.random.normal(50, 10)) / 255
    # gN2 = (np.random.rand() * 50 + 20) / 255 ** 2
    # graymodel = imGaussNoise(graymodel, gN1, gN2)
    # graymodel *= 255

    # graymodel += imblank_noise
    return graymodel, pt
