import cv2 as cv
import numpy as np
import math
import time
from GUI_Pages.IntrinsicParametersFunctions.f_fitmodel_intrinsic import f_fitmodel_intrinsic
from GUI_Pages.IntrinsicParametersFunctions.f_x_to_model_intrinsic import f_x_to_model_intrinsic

# ft = np.zeros((4,1))
# ft[0,0] = 12
# ft[1,0] = 11
# ft[2,0] = 10
# ft[3,0] = 9
# a = np.argmin(ft)
# print(a)
# exit()
                                                                                         #    10            12             14       15     16       17      18      19
# x = np.array([24.2929, 15.3788, 2.2275, 1.1504, 2.0976, 0.9458, 1.6068, 0.6413, 0.5021, 0.2908, 0.7758, 0.8985, 0.4499, 1.2952, 0.5047, 0.9540, 0.4954, 0.7056, 1.0933])
# seglen = 1.9261
#
# im_gray, pt = f_x_to_model_intrinsic(x, seglen, 0, 38, 43)
# pt = pt.astype(int)
# im_gray[pt[1,:], pt[0,:]] = 255
# cv.imwrite('test.png', im_gray)
# exit()

# image = cv.imread('big_cut_out_0.png')
image = cv.imread('temp.png')

# if image.ndim == 3:
#     image = image[...,0]

amount_of_clicks = 0

position_array = np.zeros((2,2))
# the row is the click event arranged as x, y

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')
        global amount_of_clicks
        global image
        position_array[amount_of_clicks,:] = [x,y]
        amount_of_clicks += 1
        if amount_of_clicks >= 2: cv.waitKey(1)
        # draw point on the image
        image = cv.circle(image, (x, y), 2, (0, 255, 255), -1)
        cv.imshow('Point Coordinates', image.astype(np.uint8))


cv.namedWindow('Point Coordinates')
cv.setMouseCallback('Point Coordinates', click_event)
cv.imshow('Point Coordinates', image.astype(np.uint8))
# cv.waitKey(0)
while True:
    if amount_of_clicks >= 2: break
    cv.imshow('Point Coordinates',image.astype(np.uint8))
    cv.waitKey(1)

# position_array = np.array([[27, 10],
#                            [14, 27]])

approx_centroid = (3 * position_array[0,:]  + 1 * position_array[1,:]) / (3 + 1)
heading_vec = np.array([[- (position_array[1,1] - position_array[0,1]) ],
                        [position_array[1,0] - position_array[0,0]]])

theta_0 = math.atan2(heading_vec[0], heading_vec[1])
fishlen = np.sqrt( np.sum( heading_vec ** 2  )  )
seglen = 0.09 * fishlen
x = np.zeros((19))
x[0] = approx_centroid[0]
x[1] = approx_centroid[1]
x[2] = -theta_0
x[3] = 0.95
x[4] = 1.8
x[5] = 1.2
x[6] = 1.3
x[7] = 350 / 1000
x[8] = 200 / 1000
x[9] = 120 / 1000
x[10] = .5
x[11] = .5
x[12] = .5
x[13] = .8
x[14] = .3
x[15] = .7
x[16] = .5
x[17] = .6
x[18] = 1


# seglen = 1.9261
#
# im_gray, pt = f_x_to_model_intrinsic(x, seglen, 0, 38, 43)
# cv.imwrite('test3.png', im_gray)
# exit()


x0 = np.copy(x)
# In case the image is read as RGB
if len(image.shape) > 2: image = image[...,0]
x2, fval = f_fitmodel_intrinsic(x0, seglen, image)

# im_gray = f_x_to_model_intrinsic(x, seglen, 0, size(im_real, 2), size(im_real, 1))
startTime = time.time()
im_gray, pt = f_x_to_model_intrinsic(x2, seglen, 0, image.shape[1], image.shape[0])
cv.imwrite('test.png', im_gray)


print(fishlen)




