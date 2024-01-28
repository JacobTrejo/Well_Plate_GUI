import numpy as np
import imageio
import cv2 as cv

vid = np.load('vid.npy')

greenVid = np.zeros((*(vid.shape[:2]),3,vid.shape[-1] ))

greenVid[...,1,:] = vid


np.save('greenVid.npy', greenVid)

