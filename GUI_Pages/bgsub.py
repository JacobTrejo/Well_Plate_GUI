import os

import numpy as np
import cv2

def bgsub(video):

    frames = []
    while True:
        read, frame = video.read()

        if not read:
            break
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grayImage)
    frameCount = len(frames)
    nSampFrame = min(np.fix(frameCount / 2), 100)

    #Creating a numpy array of the the sample frames
    firstSampFrame = True
    secondSampFrame = True
    for frameNum in np.fix(np.linspace(1, frameCount, int(nSampFrame))):
        if firstSampFrame:
            firstSampFrame = False
            sampFrames = frames[0]
            continue
        if secondSampFrame:
            secondSampFrame = False
            sampFrames = np.stack((sampFrames,frames[int((frameNum - 1))]),axis=0)
            continue
        sampFrames = np.vstack((sampFrames,np.array([frames[int((frameNum - 1))]])))

    sampFrames.sort(0)

    videobg = sampFrames[int(np.fix(nSampFrame * .9))]

    outputVid = []
    for frame in frames:
        # Subtract foreground from background image by allowing values beyond the 0 to 255 range of uint8
        difference_img = np.int16(videobg) - np.int16(frame)

        # Clip values in [0,255] range
        difference_img = np.clip(difference_img, 0, 255)

        # Convert difference image to uint8 for saving to video
        difference_img = np.uint8(difference_img)
        outputVid.append(difference_img)

    return outputVid

def bgsubFiles(filenames, folderName):

    frames = []
    print('the first path: ', folderName + filenames[0])
    for filename in filenames:
        frame = cv2.imread(folderName + '/' + filename)
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grayImage)

    frameCount = len(frames)
    nSampFrame = min(np.fix(frameCount / 2), 100)

    # Creating a numpy array of the the sample frames
    firstSampFrame = True
    secondSampFrame = True
    for frameNum in np.fix(np.linspace(1, frameCount, int(nSampFrame))):
        if firstSampFrame:
            firstSampFrame = False
            sampFrames = frames[0]
            continue
        if secondSampFrame:
            secondSampFrame = False
            sampFrames = np.stack((sampFrames, frames[int((frameNum - 1))]), axis=0)
            continue
        sampFrames = np.vstack((sampFrames, np.array([frames[int((frameNum - 1))]])))

    sampFrames.sort(0)

    videobg = sampFrames[int(np.fix(nSampFrame * .9))]

    outputVid = []
    for frame in frames:
        # Subtract foreground from background image by allowing values beyond the 0 to 255 range of uint8
        difference_img = np.int16(videobg) - np.int16(frame)

        # Clip values in [0,255] range
        difference_img = np.clip(difference_img, 0, 255)

        # Convert difference image to uint8 for saving to video
        difference_img = np.uint8(difference_img)
        outputVid.append(difference_img)

    return outputVid





def bgsubFolder(folderName):
    filenames = os.listdir(folderName)
    filenames.sort()

    frames = []
    for filename in filenames:
        frame = cv2.imread(folderName + '/' + filename)
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grayImage)

    frameCount = len(frames)
    nSampFrame = min(np.fix(frameCount / 2), 100)

    # Creating a numpy array of the the sample frames
    firstSampFrame = True
    secondSampFrame = True
    for frameNum in np.fix(np.linspace(1, frameCount, int(nSampFrame))):
        if firstSampFrame:
            firstSampFrame = False
            sampFrames = frames[0]
            continue
        if secondSampFrame:
            secondSampFrame = False
            sampFrames = np.stack((sampFrames, frames[int((frameNum - 1))]), axis=0)
            continue
        sampFrames = np.vstack((sampFrames, np.array([frames[int((frameNum - 1))]])))

    sampFrames.sort(0)

    videobg = sampFrames[int(np.fix(nSampFrame * .9))]

    outputVid = []
    for frame in frames:
        # Subtract foreground from background image by allowing values beyond the 0 to 255 range of uint8
        difference_img = np.int16(videobg) - np.int16(frame)

        # Clip values in [0,255] range
        difference_img = np.clip(difference_img, 0, 255)

        # Convert difference image to uint8 for saving to video
        difference_img = np.uint8(difference_img)
        outputVid.append(difference_img)

    return outputVid

# # vidObj = cv2.VideoCapture('videosActual/zebrafish.mp4')
# bgsubVid = bgsubFolder('videosActual/wellPlateImages')
# print(len(bgsubVid))





