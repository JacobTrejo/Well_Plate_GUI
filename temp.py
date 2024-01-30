import numpy as np
import imageio
import cv2 as cv

greenVid = np.load('greenVid.npy')
cv.imwrite('temp.png', greenVid[...,0])




# vid = np.load('vid.npy')
#
# greenVid = np.zeros((*(vid.shape[:2]), 3, vid.shape[-1]))
#
# greenVid[...,1,:] = vid[:,:,:]
#
# cv.imwrite('temp.png', greenVid[...,0])


# vid = np.load('vid.npy')
#
# greenVid = np.zeros((*(vid.shape[:2]),3,vid.shape[-1] ))
#
# greenVid[...,1,:] = vid
#
#
# np.save('greenVid.npy', greenVid)

