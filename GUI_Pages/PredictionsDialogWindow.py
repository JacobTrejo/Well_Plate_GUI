import numpy as np

from GUI_Pages.Auxilary import *

import torch
from GUI_Pages.NN.ResNet_Blocks_3D_four_blocks import resnet18
from GUI_Pages.NN.CustomDataset2 import CustomImageDataset
from GUI_Pages.NN.CustomDataset2 import padding
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PyQt5.QtWidgets import QApplication

import cv2 as cv
import cv2
# from VideoPlayer3 import Widget
from GUI_Pages.VideoPlayer5 import Widget
from GUI_Pages.VideoPlayerFolders2 import FolderVideoPlayer
from PyQt5 import QtCore, QtGui
import os
from GUI_Pages.bgsub import bgsub, bgsubFolder
import time
import math
from multiprocessing import Pool, Manager
from GUI_Pages.CalculateCC.evaluation_functions import evaluate_prediction

import xlsxwriter

def array2XL(array, outputPath):
    workbook = xlsxwriter.Workbook(outputPath)
    worksheet = workbook.add_worksheet()

    worksheet.write(1,0,'Frame #')
    amountOfFrames = len(array)
    for frameIdx in range(amountOfFrames):
        worksheet.write(frameIdx + 2,0, frameIdx + 1)
    amoutOfFish = array.shape[1]

    step = 2
    # Writting the title of the fish
    for fishIdx in range(amoutOfFish):
        realIdx = fishIdx * step
        worksheet.write(0, realIdx + 1, 'Fish ' + str(fishIdx + 1))
        worksheet.write(1, realIdx + 1, 'x')
        worksheet.write(1, realIdx + 2, 'y')

    for frameIdx in range(amountOfFrames):
        for fishIdx in range(amoutOfFish):
            realIdx = fishIdx * step
            x, y = array[frameIdx, fishIdx, :, 2]

            if not math.isnan(x) and not math.isnan(y):
                worksheet.write(frameIdx + 2, realIdx + 1, x)
                worksheet.write(frameIdx + 2, realIdx + 2, y)

    workbook.close()

def folderAndArray2Video(folderPath, array, outputPath):
    frameFiles = os.listdir(folderPath)
    frameFiles.sort()
    frame0 = cv.imread(folderPath + '/' + frameFiles[0])
    height, width = frame0.shape[:2]
    isGray = (frame0.ndim == 2)

    # Getting the video writter ready
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    output = cv.VideoWriter(outputPath, fourcc, int(50), (int(width), int(height)))

    amountOfFish = array.shape[1]
    amountOfFrames = len(array)
    # Saves repeating one line, probably wont speed it up too much tho
    if isGray:
        for frameIdx in range(amountOfFrames):
            frame = cv.imread(folderPath + '/' + frameFiles[frameIdx])
            rgb = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

            pts = array[frameIdx]
            for fishIdx in range(amountOfFish):
                if math.isnan(pts[fishIdx, 0, 0]): continue
                pt = pts[fishIdx].astype(int)

                # Clipping for the case that wells are near the edge
                # Perhaps skipping over them is better
                pt[0, :] = np.clip(pt[0, :], 0, width)
                pt[1, :] = np.clip(pt[1, :], 0, height)

                for ptIdx in range(10):
                    cv.circle(rgb, (pt[0,ptIdx], pt[1,ptIdx]), 2, (0,255,0), -1)
                for ptIdx in range(10, 12):
                    cv.circle(rgb, (pt[0,ptIdx], pt[1,ptIdx]), 2, (0,0,255), -1)

            output.write(rgb)
    else:
        for frameIdx in range(amountOfFrames):
            rgb = cv.imread(folderPath + '/' + frameFiles[frameIdx])

            pts = array[frameIdx]
            for fishIdx in range(amountOfFish):
                if math.isnan((pts[fishIdx, 0, 0])): continue
                pt = pts[fishIdx].astype(int)
                # Clipping for the case that wells are near the edge
                # Perhaps skipping over them is better
                pt[0,:] = np.clip(pt[0,:], 0, width)
                pt[1,:] = np.clip(pt[1,:], 0, height)
                for ptIdx in range(10):
                    cv.circle(rgb, (pt[0,ptIdx], pt[1,ptIdx]), 2, (0,255,0), -1)
                for ptIdx in range(10, 12):
                    cv.circle(rgb, (pt[0,ptIdx], pt[1,ptIdx]), 2, (0,0,255), -1)

            output.write(rgb)

    output.release()

class ProgressDialog(QDialog):

    # Output Modes
    NUMPY = 'NUMPY'
    EXCEL = 'EXCEL'
    VIDEO = 'VIDEO'

    # CC Modes
    CLIPCC = 'CLIPCC'
    SAVECC = 'SAVECC'

    def __init__(self, videoPaths, gridPath, modelPath, withCC, *args, **kwargs):
        super(ProgressDialog, self).__init__(*args, **kwargs)
        self.widthCC = withCC
        self.videoPaths = videoPaths
        self.gridPath = gridPath
        self.modelPath = modelPath

        self.outputMode = ProgressDialog.VIDEO
        self.ccMode = ProgressDialog.CLIPCC

        self.saveDirectory = None

        self.cutOffThreshold = None

        self.initUI()

    def initUI(self):
        self.setStyleSheet('background: ' + blue)
        self.setGeometry(300, 300, 550, 400)
        self.setWindowTitle('Annotating Window')
        thisWidgetsLayout = QVBoxLayout()
        self.stackedWidget = QStackedWidget()

        # The Widget for the predictions dialog
        predictionsSettingsDialog = QWidget()
        predictionsSettingsDialogLayout = QGridLayout()
        predictionsTitle = QLabel('Predict')
        predictionsTitle.setAlignment(Qt.AlignHCenter)

        modeWidget = QWidget()
        modeWidgetLayout = QVBoxLayout()
        modeLabel = QLabel('Output Mode')
        modeRadioButtonsWidget = QWidget()
        modeRadioButtonsWidgetLayout = QHBoxLayout()
        numpyButton = QRadioButton('Numpy')
        numpyButton.mode = ProgressDialog.NUMPY
        numpyButton.toggled.connect(self.onToggle)
        excelButton = QRadioButton('Excel')
        excelButton.mode = ProgressDialog.EXCEL
        excelButton.toggled.connect(self.onToggle)
        visualButton = QRadioButton('Video')
        visualButton.mode = ProgressDialog.VIDEO
        visualButton.toggled.connect(self.onToggle)
        visualButton.setChecked(True)

        modeRadioButtonsWidgetLayout.addWidget(numpyButton)
        modeRadioButtonsWidgetLayout.addWidget(excelButton)
        modeRadioButtonsWidgetLayout.addWidget(visualButton)
        modeRadioButtonsWidget.setLayout(modeRadioButtonsWidgetLayout)

        modeWidgetLayout.addWidget(modeLabel, alignment = Qt.AlignHCenter)
        modeWidgetLayout.addWidget(modeRadioButtonsWidget, alignment = Qt.AlignHCenter)

        runButtonWrapper = QWidget()
        runButtonWrapperLayout = QHBoxLayout()
        runButton = QPushButton('    Run    ')
        runButton.setStyleSheet(smallerButtonStyleSheet)
        runButton.clicked.connect(self.run)
        runButtonWrapperLayout.addWidget(runButton, alignment = Qt.AlignHCenter)
        runButtonWrapper.setLayout(runButtonWrapperLayout)


        modeWidget.setLayout(modeWidgetLayout)
        predictionsSettingsDialogLayout.addWidget(predictionsTitle, 0, 0, alignment = Qt.AlignTop)
        predictionsSettingsDialogLayout.addWidget(modeWidget, 0, 0, alignment = Qt.AlignCenter)
        predictionsSettingsDialogLayout.addWidget(runButtonWrapper, 0, 0, alignment = Qt.AlignBottom)
        predictionsSettingsDialog.setLayout(predictionsSettingsDialogLayout)

        self.stackedWidget.addWidget(predictionsSettingsDialog)

        # The widget for the correlation coefficients
        predictionsSettingsDialog = QWidget()
        predictionsSettingsDialogLayout = QGridLayout()
        predictionsTitle = QLabel('Predict With CC')
        predictionsTitle.setAlignment(Qt.AlignHCenter)

        centralWidget = QWidget()
        centralWidgetLayout = QVBoxLayout()

        modeWidget = QWidget()
        modeWidgetLayout = QVBoxLayout()
        modeLabel = QLabel('Output Mode')
        modeRadioButtonsWidget = QWidget()
        modeRadioButtonsWidgetLayout = QHBoxLayout()
        numpyButton = QRadioButton('Numpy')
        numpyButton.mode = ProgressDialog.NUMPY
        numpyButton.toggled.connect(self.onToggle)
        excelButton = QRadioButton('Excel')
        excelButton.mode = ProgressDialog.EXCEL
        excelButton.toggled.connect(self.onToggle)
        visualButton = QRadioButton('Video')
        visualButton.mode = ProgressDialog.VIDEO
        visualButton.toggled.connect(self.onToggle)
        visualButton.setChecked(True)

        modeRadioButtonsWidgetLayout.addWidget(numpyButton)
        modeRadioButtonsWidgetLayout.addWidget(excelButton)
        modeRadioButtonsWidgetLayout.addWidget(visualButton)
        modeRadioButtonsWidget.setLayout(modeRadioButtonsWidgetLayout)

        modeWidgetLayout.addWidget(modeLabel, alignment = Qt.AlignHCenter)
        modeWidgetLayout.addWidget(modeRadioButtonsWidget, alignment = Qt.AlignHCenter)
        modeWidget.setLayout(modeWidgetLayout)
        centralWidgetLayout.addWidget(modeWidget)

        ccModeWidget = QWidget()
        ccModeWidgetLayout = QVBoxLayout()
        ccModeLabel = QLabel('CC Settings')
        ccModeLabel.setAlignment(Qt.AlignHCenter)

        clipRadioButton = QRadioButton('Use CC to clip')
        clipRadioButton.mode = ProgressDialog.CLIPCC
        clipRadioButton.setChecked(True)

        self.sliderValueLabel = QLabel('  ')

        # Slider associated with the above widget
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.changeValue)
        self.slider.setEnabled(True)
        self.slider.setFixedWidth(227)
        self.slider.setValue(80)
        # self.slider.setSliderPosition()
        clipRadioButton.toggled.connect(self.ccModeToggle)

        saveRadioButton = QRadioButton('Save CC')
        saveRadioButton.mode = ProgressDialog.SAVECC
        saveRadioButton.toggled.connect(self.ccModeToggle)

        ccModeWidgetLayout.addWidget(ccModeLabel)
        ccModeWidgetLayout.addWidget(clipRadioButton)
        ccModeWidgetLayout.addWidget(self.sliderValueLabel)
        ccModeWidgetLayout.addWidget(self.slider)
        self.sliderValueLabel.move(self.slider.x(), self.slider.y() - 30)
        ccModeWidgetLayout.addWidget(saveRadioButton)
        ccModeWidget.setLayout(ccModeWidgetLayout)
        centralWidgetLayout.addWidget(ccModeWidget)

        centralWidget.setLayout(centralWidgetLayout)

        runButtonWrapper = QWidget()
        runButtonWrapperLayout = QHBoxLayout()
        runButton = QPushButton('    Run    ')
        runButton.setStyleSheet(smallerButtonStyleSheet)
        runButton.clicked.connect(self.runWithCC)
        runButtonWrapperLayout.addWidget(runButton, alignment = Qt.AlignHCenter)
        runButtonWrapper.setLayout(runButtonWrapperLayout)


        predictionsSettingsDialogLayout.addWidget(predictionsTitle, 0, 0, alignment = Qt.AlignTop)
        predictionsSettingsDialogLayout.addWidget(centralWidget, 0, 0, alignment = Qt.AlignCenter)
        predictionsSettingsDialogLayout.addWidget(runButtonWrapper, 0, 0, alignment = Qt.AlignBottom)
        predictionsSettingsDialog.setLayout(predictionsSettingsDialogLayout)

        self.stackedWidget.addWidget(predictionsSettingsDialog)

        # The Widget in charge of showing the progress
        progressWidget = QWidget()
        progressWidgetLayout = QGridLayout()

        # This is to help center it
        titleLabelWrapper = QWidget()
        titleLabelWrapperLayout = QHBoxLayout()
        titleLabel = QLabel('Progress')
        titleLabelWrapperLayout.addWidget(titleLabel, alignment = Qt.AlignHCenter)
        titleLabelWrapper.setLayout(titleLabelWrapperLayout)

        # This widget will contain the progress bar
        # and the label information
        centralWidget = QWidget()
        centralWidgetLayout = QVBoxLayout()
        self.progressLabel = QLabel('Analyzing Video 1 of ' + str(len(self.videoPaths)))
        self.progressBar = QProgressBar()
        self.progressBar.setFixedWidth(350)

        centralWidgetLayout.addWidget(self.progressLabel, alignment = Qt.AlignHCenter)
        centralWidgetLayout.addWidget(self.progressBar)
        centralWidget.setLayout(centralWidgetLayout)

        progressWidgetLayout.addWidget(titleLabelWrapper, 0, 0, alignment = Qt.AlignTop)
        progressWidgetLayout.addWidget(centralWidget, 0, 0, alignment = Qt.AlignCenter )

        progressWidget.setLayout(progressWidgetLayout)

        self.stackedWidget.addWidget(predictionsSettingsDialog)
        self.stackedWidget.addWidget(progressWidget)

        if self.widthCC:
            self.stackedWidget.setCurrentIndex(1)
        else:
            self.stackedWidget.setCurrentIndex(0)

        thisWidgetsLayout.addWidget(self.stackedWidget, 1)

        self.setLayout(thisWidgetsLayout)

    def onToggle(self):
        radioButton = self.sender()
        self.outputMode = radioButton.mode

    def ccModeToggle(self):
        radioButton = self.sender()
        self.ccMode = radioButton.mode
        self.slider.setEnabled(self.ccMode == ProgressDialog.CLIPCC)

    def changeValue(self, value):
        valueRange = 100
        fraction = (value) / valueRange
        self.cutOffThreshold = round(fraction, 2)
        self.sliderValueLabel.setText(str(round(fraction, 2)))
        width = self.slider.width()
        offset = int(((value) / valueRange) * width)

        # self.label = QLabel(self)
        # self.label.setText(str(value))

        # self.sliderValueLabel.setGeometry(self.slider.x() + offset, self.slider.y() - 1, 100, 50)
        self.sliderValueLabel.setContentsMargins(self.slider.x() + offset - 20, 0, 0, 0)
        # self.label.setGeometry(self.slider.x() + offset, self.slider.y() - 1, 100, 50)
        #self.label.move(self.slider.x() + offset, self.slider.y() - 1)

        # self.update()
    def run(self):
        self.stackedWidget.setCurrentIndex(2)

        # Asking for a directory in which the files will be saved in
        self.saveDirectory = str(QFileDialog.getExistingDirectory(self, "Select Directory To Save Your Data:"))
        # TODO: add safety

        print('You switched the widget')
        time.sleep(1)

        grid = np.load(self.gridPath)
        videoPaths = self.videoPaths

        # # Fast testing version
        # self.gridPath = 'grids/wellplate.npy'
        # grid = np.load(self.gridPath)
        # # self.modelPath = 'models/resnet_pose_best_python_230608_four_blocks.pt'
        # self.modelPath = 'models/resnet_pose.pt'
        # videoPaths = ['videos/wellPlateImages']

        # Let's expand the grid to the size it should be
        if os.path.isfile(videoPaths[0]):
            # Lets assume that it is a video
            try:
                vid = cv.VideoCapture(videoPaths[0])
                width = vid.get(cv.CAP_PROP_FRAME_WIDTH)
                grid *= width
            except:
                print('One of your files was invalid')
        elif os.path.isdir(videoPaths[0]):
            # We are assuming that it is a folder with images
            try:
                imageNames = os.listdir(videoPaths[0])
                frame0 = cv.imread(videoPaths[0] + '/' + imageNames[0])
                width = frame0.shape[1]
                grid *= width
            except:
                print('One of your folders was invalid')
        else:
            # Will we even reach this line ??
            print('One of the videos you have selected is not a valid format')

        # Making sure the radius of the well is not too big to cause a shift
        grid[:, 2] = np.clip(grid[:, 2], 0, 49)

        # Let's load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnetModel = resnet18(1, 12, activation='leaky_relu').to(self.device)
        self.resnetModel = nn.DataParallel(self.resnetModel)
        if torch.cuda.is_available():
            self.resnetModel.load_state_dict(torch.load(self.modelPath))
        else:
            self.resnetModel.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu')))

        self.resnetModel.eval()
        torch.no_grad()
        n_cuda = torch.cuda.device_count()
        if (torch.cuda.is_available()):
            print(str(n_cuda) + 'GPUs are available!')
            self.nworkers = n_cuda * 12
            self.pftch_factor = 2
        else:
            print('Cuda is not available. Training without GPUs. This might take long')
            self.nworkers = 2
            self.pftch_factor = 1
            # self.nworkers = 1
            # self.pftch_factor = 1
        self.batch_size = 512 * n_cuda
        if n_cuda == 0: self.batch_size = 10
        # Initialize
        for videoPath in videoPaths:
            if os.path.isfile(videoPath):
                self.predict4VideoFile(videoPath, grid)
            elif os.path.isdir(videoPath):
                self.predict4Folder(videoPath, grid)

    def predict4VideoFile(self, videoPath, grid):
        print('You predicted for the video')

    def predict4Folder(self, folderPath, grid):
        fileFolderSplit = folderPath.split('/')[-1]
        filefolderName = fileFolderSplit.split('.')[0]

        bgsubList = bgsubFolder(folderPath)
        frame0 = bgsubList[0]
        # self.predictForFrame(frame0, grid)
        # self.predictForFrames(bgsubList[:10], grid)
        amountOfImages = len(bgsubList)
        amountOfCircles = len(grid)
        fishData = np.zeros((amountOfImages, amountOfCircles, 2, 12))
        # Analyzing the data in batches
        batchsize = 100
        amountOfBatches = int(np.ceil(amountOfImages / batchsize))

        self.progressBar.setRange(0, amountOfImages - 1)
        for batchIdx in range(amountOfBatches - 1):
            startingFrame = batchIdx * batchsize
            endingFrame = (batchIdx + 1) * batchsize
            fishDataInstance = self.predictForFrames(bgsubList[startingFrame:endingFrame], grid)
            fishData[startingFrame: endingFrame] = fishDataInstance

            self.progressBar.setValue(endingFrame)
            self.update()
            QApplication.processEvents()
        # The final batch
        startingFrame = (amountOfBatches - 1) * batchsize
        fishDataInstance = self.predictForFrames(bgsubList[startingFrame:], grid)
        fishData[startingFrame:] = fishDataInstance
        self.progressBar.setValue(amountOfImages)
        self.progressLabel.setText('Done')
        self.update()
        QApplication.processEvents()


        # Saving the data appropriately
        if self.outputMode == ProgressDialog.NUMPY:
            np.save(self.saveDirectory + '/' + filefolderName + '.npy', fishData)
        elif self.outputMode == ProgressDialog.VIDEO:
            folderAndArray2Video(folderPath, fishData, self.saveDirectory + '/' + filefolderName + '.avi' )
        elif self.outputMode == ProgressDialog.EXCEL:
            array2XL(fishData, self.saveDirectory + '/' + filefolderName + '.xlsx')

    def predictForFrames(self, images, grid):

        green = [0, 255, 0]
        red = [0, 0, 255]
        rgbs = []
        rgb = np.stack((images[0], images[0], images[0]), axis=2)
        cutOutList = []
        circIdx = 0
        imageIdx = 0
        amountOfCircles = grid.shape[0]
        amountOfImages = len(images)

        # NOTE: you might just want shape amountOfImages, amountOfCircles, 2 for COM
        fishData = np.zeros((amountOfImages, amountOfCircles, 2, 12))

        for image in images:
            for circ in grid:
                center = (int(circ[0]), int(circ[1]))
                radius = int(circ[2])

                sX = center[0] - radius
                bX = center[0] + radius
                sY = center[1] - radius
                bY = center[1] + radius

                cutOut = image[sY:bY + 1, sX:bX + 1]

                cutOut = cutOut.astype(float)
                cutOut *= 255 / np.max(cutOut)
                cutOut = cutOut.astype(np.uint8)

                cutOutList.append(cutOut)

        # #Testing
        # for cutoutIdx, cutout in enumerate(cutOutList):
        #     cv.imwrite('outputs/cutouts/cutout_' + str(cutoutIdx) + '.png', cutout)
        model = self.resnetModel

        # create a quantized model instance
        model_int8 = torch.ao.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear, nn.Conv2d},  # a set of layers to dynamically quantize
            dtype=torch.qint8)

        print('starting')
        start = time.time()
        transform = transforms.Compose([padding(), transforms.PILToTensor()])
        data = CustomImageDataset(cutOutList, transform=transform)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.nworkers,
                            prefetch_factor=self.pftch_factor, persistent_workers=True)

        for i, im in enumerate(loader):
            im = im.to(self.device)
            # pose_recon = model_int8(im)
            pose_recon = model(im)

            # pose_recon = pose_recon.detach().cpu().numpy()
            # im = np.squeeze(im.detach().cpu().numpy())

            # The following is extra computation stuff lets take it out for now
            pose_recon = pose_recon.detach().cpu().numpy()
            im = np.squeeze(im.cpu().detach().numpy(), axis=1)
            for imIdx in range(im.shape[0]):
                im1 = im[imIdx, ...]
                im1 *= 255
                im1 = im1.astype(np.uint8)
                pt1 = pose_recon[imIdx, ...]

                noFishThreshold = 10
                if np.max(pt1) < noFishThreshold or np.max(im1) <= 0:
                    fishData[imageIdx, circIdx, ...] = np.nan
                else:

                    # pt1 = pt1.astype(int)
                    # im1[pt1[1,:], pt1[0,:]] =  255
                    # cv.imwrite('test.png', im1)
                    # exit()

                    # Fix this part up, should try to get rid of using np.where
                    nonZero = np.where(im1 > 0)
                    sY = np.min(nonZero[0])
                    sX = np.min(nonZero[1])
                    pt1[0, :] -= sX
                    pt1[1, :] -= sY

                    circ = grid[circIdx]
                    center = (int(circ[0]), int(circ[1]))
                    radius = int(circ[2])

                    sX = center[0] - radius
                    bX = center[0] + radius
                    sY = center[1] - radius
                    bY = center[1] + radius
                    # sX, sY, bX, bY = boxes[ imIdx, ...]
                    pt1[0, :] += sX
                    pt1[1, :] += sY

                    # For visualizing purposes
                    # pt1 = pt1.astype(int)
                    #
                    # rgb[pt1[1, :10], pt1[0, :10]] = green
                    # rgb[pt1[1, 10:], pt1[0, 10:]] = red

                    fishData[imageIdx, circIdx, ...] = pt1

                circIdx += 1
                if circIdx == amountOfCircles:
                    circIdx = 0
                    imageIdx += 1
                    if imageIdx == len(images):
                        print('You saved the data')

                        # dlg = QFileDialog()
                        # fileNameForData = dlg.getSaveFileName()
                        # np.save(fileNameForData[0], fishData)




                        end = time.time()
                        print("Finished predicting")
                        print('duration: ', end - start)
                        return fishData



    def runWithCC(self):
        self.stackedWidget.setCurrentIndex(2)

        # Asking for a directory in which the files will be saved in
        self.saveDirectory = str(QFileDialog.getExistingDirectory(self, "Select Directory To Save Your Data:"))
        # TODO: add safety

        print('You switched the widget')
        time.sleep(1)

        grid = np.load(self.gridPath)
        videoPaths = self.videoPaths

        # # Fast testing version
        # self.gridPath = 'grids/wellplate.npy'
        # grid = np.load(self.gridPath)
        # # self.modelPath = 'models/resnet_pose_best_python_230608_four_blocks.pt'
        # self.modelPath = 'models/resnet_pose.pt'
        # videoPaths = ['videos/wellPlateImages']

        # Let's expand the grid to the size it should be
        if os.path.isfile(videoPaths[0]):
            # Lets assume that it is a video
            try:
                vid = cv.VideoCapture(videoPaths[0])
                width = vid.get(cv.CAP_PROP_FRAME_WIDTH)
                grid *= width
            except:
                print('One of your files was invalid')
        elif os.path.isdir(videoPaths[0]):
            # We are assuming that it is a folder with images
            try:
                imageNames = os.listdir(videoPaths[0])
                frame0 = cv.imread(videoPaths[0] + '/' + imageNames[0])
                width = frame0.shape[1]
                grid *= width
            except:
                print('One of your folders was invalid')
        else:
            # Will we even reach this line ??
            print('One of the videos you have selected is not a valid format')

        # Making sure the radius of the well is not too big to cause a shift
        grid[:, 2] = np.clip(grid[:, 2], 0, 49)

        # Let's load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnetModel = resnet18(1, 12, activation='leaky_relu').to(self.device)
        self.resnetModel = nn.DataParallel(self.resnetModel)
        if torch.cuda.is_available():
            self.resnetModel.load_state_dict(torch.load(self.modelPath))
        else:
            self.resnetModel.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu')))

        self.resnetModel.eval()
        torch.no_grad()
        n_cuda = torch.cuda.device_count()
        if (torch.cuda.is_available()):
            print(str(n_cuda) + 'GPUs are available!')
            self.nworkers = n_cuda * 12
            self.pftch_factor = 2
        else:
            print('Cuda is not available. Training without GPUs. This might take long')
            self.nworkers = 2
            self.pftch_factor = 1
            # self.nworkers = 1
            # self.pftch_factor = 1
        self.batch_size = 512 * n_cuda
        if n_cuda == 0: self.batch_size = 10
        # Initialize
        for videoPath in videoPaths:
            if os.path.isfile(videoPath):
                self.predict4VideoFileWithCC(videoPath, grid)
            elif os.path.isdir(videoPath):
                self.predict4FolderWithCC(videoPath, grid)

    def predict4VideoFileWithCC(self, videoPath, grid):
        print('You predicted for the video')

    def predict4FolderWithCC(self, folderPath, grid):
        fileFolderSplit = folderPath.split('/')[-1]
        filefolderName = fileFolderSplit.split('.')[0]

        bgsubList = bgsubFolder(folderPath)
        frame0 = bgsubList[0]
        # self.predictForFrame(frame0, grid)
        # self.predictForFrames(bgsubList[:10], grid)

        amountOfFrames = len(bgsubList)
        amountOfCircles = len(grid)
        self.progressBar.setRange(0, amountOfFrames - 1)
        fishData = np.zeros((amountOfFrames, amountOfCircles, 2, 12))
        ccData = np.zeros((amountOfFrames, amountOfCircles))

        batchsize = 100
        amountOfBatches = int(np.ceil(amountOfFrames/batchsize))
        for batchIdx in range(amountOfBatches - 1):
            startingFrame = batchIdx * batchsize
            endingFrame = (batchIdx + 1) * batchsize

            if self.ccMode == ProgressDialog.CLIPCC:
                fishDataInstance, ccDataInstance = self.predictForFramesWithCCThreshold(bgsubList[startingFrame: endingFrame], grid)
            else:
                fishDataInstance, ccDataInstance = self.predictForFramesWithCCNoThreshold(bgsubList[startingFrame: endingFrame], grid)

            fishData[startingFrame:endingFrame] = fishDataInstance
            ccData[startingFrame:endingFrame] = ccDataInstance

            self.progressBar.setValue(endingFrame)
            self.update()
            QApplication.processEvents()

        startingFrame = (amountOfBatches - 1) * batchsize
        if self.ccMode == ProgressDialog.CLIPCC:
            fishDataInstance, ccDataInstance = self.predictForFramesWithCCThreshold(
                bgsubList[startingFrame: ], grid)
        else:
            fishDataInstance, ccDataInstance = self.predictForFramesWithCCNoThreshold(
                bgsubList[startingFrame: ], grid)
        fishData[startingFrame:] = fishDataInstance
        ccData[startingFrame:] = ccDataInstance
        self.progressBar.setValue(amountOfFrames)
        self.progressLabel.setText('Done')
        self.update()
        QApplication.processEvents()


        # Saving the data appropriately
        if self.outputMode == ProgressDialog.NUMPY:
            np.save(self.saveDirectory + '/' + filefolderName + '.npy', fishData)
            np.save(self.saveDirectory + '/' + filefolderName + '_cc.npy', ccData)
        elif self.outputMode == ProgressDialog.VIDEO:
            folderAndArray2Video(folderPath, fishData, self.saveDirectory + '/' + filefolderName + '.avi')
            np.save(self.saveDirectory + '/' + filefolderName + '_cc.npy', ccData)
        elif self.outputMode == ProgressDialog.EXCEL:
            array2XL(fishData, self.saveDirectory + '/' + filefolderName + '.xlsx')

    def predictForFramesWithCCThreshold(self, images, grid):

        green = [0, 255, 0]
        red = [0, 0, 255]
        rgbs = []
        rgb = np.stack((images[0], images[0], images[0]), axis=2)
        cutOutList = []
        circIdx = 0
        imageIdx = 0
        amountOfCircles = grid.shape[0]
        amountOfImages = len(images)
        # self.progressBar.setRange(0, amountOfImages - 1)
        # NOTE: you might just want shape amountOfImages, amountOfCircles, 2 for COM
        fishData = np.zeros((amountOfImages, amountOfCircles, 2, 12))
        ccFishData = np.zeros((amountOfImages, amountOfCircles))

        for image in images:
            for circ in grid:
                center = (int(circ[0]), int(circ[1]))
                radius = int(circ[2])

                sX = center[0] - radius
                bX = center[0] + radius
                sY = center[1] - radius
                bY = center[1] + radius

                cutOut = image[sY:bY + 1, sX:bX + 1]

                cutOut = cutOut.astype(float)
                cutOut *= 255 / np.max(cutOut)
                cutOut = cutOut.astype(np.uint8)

                cutOutList.append(cutOut)

        # #Testing
        # for cutoutIdx, cutout in enumerate(cutOutList):
        #     cv.imwrite('outputs/cutouts/cutout_' + str(cutoutIdx) + '.png', cutout)
        model = self.resnetModel

        # create a quantized model instance
        model_int8 = torch.ao.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear, nn.Conv2d},  # a set of layers to dynamically quantize
            dtype=torch.qint8)

        print('starting')
        start = time.time()
        transform = transforms.Compose([padding(), transforms.PILToTensor()])
        data = CustomImageDataset(cutOutList, transform=transform)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.nworkers,
                            prefetch_factor=self.pftch_factor, persistent_workers=True)

        for i, im in enumerate(loader):
            im = im.to(self.device)
            # pose_recon = model_int8(im)
            pose_recon = model(im)

            # pose_recon = pose_recon.detach().cpu().numpy()
            # im = np.squeeze(im.detach().cpu().numpy())

            # The following is extra computation stuff lets take it out for now
            pose_recon = pose_recon.detach().cpu().numpy()
            im = np.squeeze(im.cpu().detach().numpy(), axis=1)
            for imIdx in range(im.shape[0]):
                im1 = im[imIdx, ...]
                im1 *= 255
                im1 = im1.astype(np.uint8)
                pt1 = pose_recon[imIdx, ...]

                noFishThreshold = 10
                if np.max(pt1) < noFishThreshold or np.max(im1) <= 0:
                    fishData[imageIdx, circIdx, ...] = np.nan
                    ccFishData[imageIdx, circIdx, ...] = np.nan
                else:
                    cc, _ = evaluate_prediction(im1, pt1)
                    if cc > self.cutOffThreshold:

                        # pt1 = pt1.astype(int)
                        # im1[pt1[1,:], pt1[0,:]] =  255
                        # cv.imwrite('test.png', im1)
                        # exit()

                        # Fix this part up, should try to get rid of using np.where
                        nonZero = np.where(im1 > 0)
                        sY = np.min(nonZero[0])
                        sX = np.min(nonZero[1])
                        pt1[0, :] -= sX
                        pt1[1, :] -= sY

                        circ = grid[circIdx]
                        center = (int(circ[0]), int(circ[1]))
                        radius = int(circ[2])

                        sX = center[0] - radius
                        bX = center[0] + radius
                        sY = center[1] - radius
                        bY = center[1] + radius
                        # sX, sY, bX, bY = boxes[ imIdx, ...]
                        pt1[0, :] += sX
                        pt1[1, :] += sY

                        # For visualizing purposes
                        # pt1 = pt1.astype(int)
                        #
                        # rgb[pt1[1, :10], pt1[0, :10]] = green
                        # rgb[pt1[1, 10:], pt1[0, 10:]] = red

                        fishData[imageIdx, circIdx, ...] = pt1
                        ccFishData[imageIdx, circIdx] = cc
                    else:
                        fishData[imageIdx, circIdx, ...] = np.nan
                        ccFishData[imageIdx, circIdx] = np.nan

                circIdx += 1
                if circIdx == amountOfCircles:
                    circIdx = 0
                    imageIdx += 1
                    if imageIdx == len(images):
                        print('You saved the data')

                        # dlg = QFileDialog()
                        # fileNameForData = dlg.getSaveFileName()
                        # np.save(fileNameForData[0], fishData)

                        end = time.time()
                        print("Finished predicting")
                        print('duration: ', end - start)
                        # self.progressLabel.setText('Done')
                        # QApplication.processEvents()
                        return fishData, ccFishData

            # self.progressBar.setValue(imageIdx)
            # self.update()
            # QApplication.processEvents()

    def predictForFramesWithCCNoThreshold(self, images, grid):

        green = [0, 255, 0]
        red = [0, 0, 255]
        rgbs = []
        rgb = np.stack((images[0], images[0], images[0]), axis=2)
        cutOutList = []
        circIdx = 0
        imageIdx = 0
        amountOfCircles = grid.shape[0]
        amountOfImages = len(images)
        # self.progressBar.setRange(0, amountOfImages - 1)
        # NOTE: you might just want shape amountOfImages, amountOfCircles, 2 for COM
        fishData = np.zeros((amountOfImages, amountOfCircles, 2, 12))
        ccData = np.zeros((amountOfImages, amountOfCircles))

        for image in images:
            for circ in grid:
                center = (int(circ[0]), int(circ[1]))
                radius = int(circ[2])

                sX = center[0] - radius
                bX = center[0] + radius
                sY = center[1] - radius
                bY = center[1] + radius

                cutOut = image[sY:bY + 1, sX:bX + 1]

                cutOut = cutOut.astype(float)
                cutOut *= 255 / np.max(cutOut)
                cutOut = cutOut.astype(np.uint8)

                cutOutList.append(cutOut)

        # #Testing
        # for cutoutIdx, cutout in enumerate(cutOutList):
        #     cv.imwrite('outputs/cutouts/cutout_' + str(cutoutIdx) + '.png', cutout)
        model = self.resnetModel

        # create a quantized model instance
        model_int8 = torch.ao.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear, nn.Conv2d},  # a set of layers to dynamically quantize
            dtype=torch.qint8)

        print('starting')
        start = time.time()
        transform = transforms.Compose([padding(), transforms.PILToTensor()])
        data = CustomImageDataset(cutOutList, transform=transform)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.nworkers,
                            prefetch_factor=self.pftch_factor, persistent_workers=True)
        ccList = []
        for i, im in enumerate(loader):
            im = im.to(self.device)
            # pose_recon = model_int8(im)
            pose_recon = model(im)

            # pose_recon = pose_recon.detach().cpu().numpy()
            # im = np.squeeze(im.detach().cpu().numpy())

            # The following is extra computation stuff lets take it out for now
            pose_recon = pose_recon.detach().cpu().numpy()
            im = np.squeeze(im.cpu().detach().numpy(), axis=1)
            for imIdx in range(im.shape[0]):
                im1 = im[imIdx, ...]
                im1 *= 255
                im1 = im1.astype(np.uint8)
                pt1 = pose_recon[imIdx, ...]

                noFishThreshold = 10
                if np.max(pt1) < noFishThreshold or np.max(im1) <= 0:
                    fishData[imageIdx, circIdx, ...] = np.nan
                    ccData[imageIdx, circIdx] = np.nan
                else:
                    ccList.append((im1, pt1, imageIdx, circIdx))
                    cc, _ = evaluate_prediction(im1, pt1)

                    # pt1 = pt1.astype(int)
                    # im1[pt1[1,:], pt1[0,:]] =  255
                    # cv.imwrite('test.png', im1)
                    # exit()

                    # Fix this part up, should try to get rid of using np.where
                    nonZero = np.where(im1 > 0)
                    sY = np.min(nonZero[0])
                    sX = np.min(nonZero[1])
                    pt1[0, :] -= sX
                    pt1[1, :] -= sY

                    circ = grid[circIdx]
                    center = (int(circ[0]), int(circ[1]))
                    radius = int(circ[2])

                    sX = center[0] - radius
                    bX = center[0] + radius
                    sY = center[1] - radius
                    bY = center[1] + radius
                    # sX, sY, bX, bY = boxes[ imIdx, ...]
                    pt1[0, :] += sX
                    pt1[1, :] += sY

                    # For visualizing purposes
                    # pt1 = pt1.astype(int)
                    #
                    # rgb[pt1[1, :10], pt1[0, :10]] = green
                    # rgb[pt1[1, 10:], pt1[0, 10:]] = red

                    fishData[imageIdx, circIdx, ...] = pt1
                    ccData[imageIdx, circIdx] = cc

                circIdx += 1
                if circIdx == amountOfCircles:
                    circIdx = 0
                    imageIdx += 1
                    if imageIdx == len(images):
                        print('You saved the data')

                        # dlg = QFileDialog()
                        # fileNameForData = dlg.getSaveFileName()
                        # np.save(fileNameForData[0], fishData)

                        end = time.time()
                        print("Finished predicting")
                        print('duration: ', end - start)
                        # self.progressLabel.setText('Done')
                        # QApplication.processEvents()
                        return fishData, ccData

            # self.progressBar.setValue(imageIdx)
            # self.update()
            # QApplication.processEvents()





