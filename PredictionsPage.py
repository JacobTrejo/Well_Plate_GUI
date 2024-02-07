from Auxilary import *

import torch
from NN.ResNet_Blocks_3D_four_blocks import resnet18
from NN.CustomDataset2 import CustomImageDataset
from NN.CustomDataset2 import padding
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import cv2 as cv
import cv2
# from VideoPlayer3 import Widget
from VideoPlayer5 import Widget
from PyQt5 import QtCore, QtGui
import os
from bgsub import bgsub, bgsubFolder
import time

class ImageViewer(QLabel):
    pixmap = None
    _sizeHint = QSize()
    ratio = Qt.KeepAspectRatio
    transformation = Qt.SmoothTransformation

    def __init__(self, pixmap=None):
        super().__init__()
        self.setPixmap(pixmap)

    def setPixmap(self, pixmap):
        if self.pixmap != pixmap:
            self.pixmap = pixmap
            if isinstance(pixmap, QPixmap):
                self._sizeHint = pixmap.size()
            else:
                self._sizeHint = QSize()
            self.updateGeometry()
            self.updateScaled()

    def setAspectRatio(self, ratio):
        if self.ratio != ratio:
            self.ratio = ratio
            self.updateScaled()

    def setTransformation(self, transformation):
        if self.transformation != transformation:
            self.transformation = transformation
            self.updateScaled()

    def updateScaled(self):
        if self.pixmap:
            self.scaled = self.pixmap.scaled(self.size(), self.ratio, self.transformation)
        self.update()

    def sizeHint(self):
        return self._sizeHint

    def resizeEvent(self, event):
        self.updateScaled()

    def paintEvent(self, event):
        if not self.pixmap:
            super().paintEvent(event)
            return
        qp = QPainter(self)
        r = self.scaled.rect()
        r.moveCenter(self.rect().center())
        qp.drawPixmap(r, self.scaled)

class ScrollLabel(QLabel):
    def __init__(self, *args, **kwargs):
        self.path = None
        super(ScrollLabel, self).__init__(*args, **kwargs)

class TitleLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super(TitleLabel, self).__init__( *args, **kwargs)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    def resizeEvent(self, a0):
        font = self.font()
        font.setPixelSize( int( self.height() * .7))
        self.setFont(font)

class DrawingWidget(QLabel):

    def __init__(self, *args, **kwargs):
        super(DrawingWidget, self).__init__(*args, **kwargs)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setPen(Qt.magenta)
        center = QtCore.QPoint(0, 0)
        qp.drawEllipse(center, 10, 10)

class PredictionPage(QWidget):

    def __init__(self, *args, **kwargs):
        super(PredictionPage, self).__init__(*args, **kwargs)

        # Temp ?
        self.drawingItems = []

        self.gridPath = None
        self.modelPath = None

        self.videoPathList = []

        self.videoList = []
        self.selectedPath = []

        self.setStyleSheet('background: ' + blue + ';')

        self.frame1 = QFrame()
        layout1 = QVBoxLayout()
        # The elements of the first frame
        label1 = QLabel('Selected Videos:')
        label1.setStyleSheet('border: 0px')
        label1.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.scroll1 = QScrollArea(self.frame1)
        # scroll1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        button1 = QPushButton('Add', self)
        button1.setStyleSheet(smallerButtonStyleSheet)
        button1.clicked.connect(self.getVideos)
        self.vboxForScrollArea = QVBoxLayout()

        # for idx in range(5):
        #     temp = QLabel('temp', scroll1)
        #     temp.mousePressEvent = self.pressed
        #     self.vboxForScrollArea.addWidget(temp)

            #scroll1.addScrollBarWidget(temp, Qt.AlignmentFlag.AlignLeft)
        self.verticalWidgetForScrollArea = QWidget()
        self.verticalWidgetForScrollArea.setLayout(self.vboxForScrollArea)
        self.verticalWidgetForScrollArea.setStyleSheet('border: 0px')
        # temp2 = QWidget()
        # hbox = QHBoxLayout()
        # hbox.addWidget(self.verticalWidgetForScrollArea)
        # temp2.setLayout(hbox)
        self.scroll1.setWidget(self.verticalWidgetForScrollArea)
        self.scroll1.setWidgetResizable(True)
        # scroll1.setLayout(vbox)
        # scroll1.setLayout(QGridLayout())
        # scroll1.setLayout(vbox)

        # The adding the widgets to the layout
        layout1.addWidget(label1)
        layout1.addWidget(self.scroll1)
        layout1.addWidget(button1)
        layout1.setStretch(1,1)
        self.frame1.setLayout(layout1)
        self.frame1.setStyleSheet('border: 1px solid black')


        self.frame2 = QFrame()
        layout2 = QVBoxLayout()
        # The elements of the second frame
        titleLabel2 = QLabel('Selected Grid')
        titleLabel2.setStyleSheet('border: 0px')
        titleLabel2.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.gridLabel = QLabel('No Grid Selected')
        self.gridLabel.mousePressEvent = self.pressedGridLabel
        self.gridLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gridLabel.setStyleSheet('color: ' + firstColorName)
        # self.gridLabel.mousePressEvent = self.pressed
        # bottomLabel = QLabel('Select Grid')
        button2 = QPushButton('Select Grid', self)
        button2.setStyleSheet(smallerButtonStyleSheet)
        button2.clicked.connect(self.getGrid)
        # adding the elements to the second frame
        layout2.addWidget(titleLabel2)
        layout2.addWidget(self.gridLabel)
        layout2.addWidget(button2)
        layout2.setStretch(1,1)
        self.frame2.setLayout(layout2)
        self.frame2.setStyleSheet('border: 1px solid black')


        self.frame3 = QFrame()
        layout3 = QVBoxLayout()
        # The elements of the third frame
        titleLabel3 = QLabel('Selected Model', self)
        titleLabel3.setStyleSheet('border: 0px')
        titleLabel3.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.modelLabel = QLabel('No Model Selected')
        self.modelLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.modelLabel.setStyleSheet('color: ' +firstColorName)
        button3 = QPushButton('Selected Model')
        button3.setStyleSheet(smallerButtonStyleSheet)
        button3.clicked.connect(self.getModels)
        # adding the elements to the third frame
        layout3.addWidget(titleLabel3)
        layout3.addWidget(self.modelLabel)
        layout3.addWidget(button3)
        layout3.setStretch(1,1)
        self.frame3.setLayout(layout3)
        self.frame3.setStyleSheet('border: 1px solid black')

        self.frame4 = QFrame()
        # The elements of the fourth frame
        button4 = QPushButton('Run', self)
        button4.setStyleSheet(smallerButtonStyleSheet)
        button4.clicked.connect(self.run)
        # adding the elements to the fourth frame

        ccButton = QPushButton('Calculate CC')
        ccButton.setStyleSheet(smallerButtonStyleSheet)

        layout4 = QVBoxLayout()
        layout4.addWidget(button4)
        layout4.setStretch(0,1)
        layout4.addWidget(ccButton)

        self.frame4.setLayout(layout4)
        self.frame4.setStyleSheet('border: 1px solid black')
        self.setMinimumSize(854, 510)
        self.initUI()

    def initUI(self):
        rightLayout = QVBoxLayout()
        rightLayout.setContentsMargins(0,0,0,0)
        rightLayout.addWidget(self.frame1)
        rightLayout.addWidget(self.frame2)
        rightLayout.addWidget(self.frame3)
        rightLayout.addWidget(self.frame4)
        rightLayout.setStretch(0,1)
        rightLayout.setStretch(1,1)
        rightLayout.setStretch(2,1)
        rightLayout.setStretch(3,1)
        rightWidget = QWidget()
        rightWidget.setLayout(rightLayout)
        rightWidget.setStyleSheet(
            'margin-top: 0px;' +
            'margin-bottom: 0px;' +
            'padding: 0px 0'
        )

        # label = QLabel('Hello')
        # self.label = ImageViewer(QPixmap('wellplate.png'))
        self.label = ImageViewer()

        # label.setPixmap(QPixmap('wellplate.png'))

        # print(label.size())
        leftWidget = QWidget()
        # leftWidgetLayout = QGridLayout()
        leftWidgetLayout = QVBoxLayout()
        leftWidgetLayout.setContentsMargins(0,0,0,0)
        leftWidgetLayout.setSpacing(0)
        self.label.setStyleSheet('border: 1px solid black;' +
                                 'color: white;')
        self.label.setText('No Video Selected')
        self.label.setStyleSheet('border: 1px solid')
        self.label.setAlignment(Qt.AlignCenter)

        drawingWidget = DrawingWidget()
        # NOTE: We are testing the widget here
        self.widget = Widget()
        # tempWidget = QWidget()
        # tempWidgetLayout = QHBoxLayout()
        # tempLeftSpacer = QWidget()
        # tempRightSpacer = QWidget()
        # tempWidgetLayout.addWidget(tempLeftSpacer, 0)
        # tempWidgetLayout.addWidget(self.widget, 1)
        # tempWidgetLayout.addWidget(tempRightSpacer, 0)
        # tempWidget.setLayout(tempWidgetLayout)

        # leftWidgetLayout.addWidget(drawingWidget, 0, 0, 1, 1)
        # leftWidgetLayout.addWidget(self.label, 0, 0, 1, 1)
        leftWidgetLayout.addWidget(self.widget, 1)

        leftWidgetBottomBar = QWidget()
        leftWidgetBottomBar.setStyleSheet('')
        leftWidgetBottomBarLayout = QHBoxLayout()
        leftWidgetBottomBar.setContentsMargins(0,0,0,0)
        playButton = QPushButton('play')
        playButton.clicked.connect(self.widget.toggledPlay)
        slider = QSlider(Qt.Horizontal)
        self.widget.connectSlider(slider)
        leftWidgetBottomBarLayout.addWidget(playButton)
        leftWidgetBottomBarLayout.addWidget(slider)
        leftWidgetBottomBar.setLayout(leftWidgetBottomBarLayout)

        leftWidgetLayout.addWidget(leftWidgetBottomBar, 0)

        leftWidget.setLayout(leftWidgetLayout)
        leftWidget.setStyleSheet('border: 1px solid black')
        # self.label.show()

        # self.label.adjustSize()
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(leftWidget)
        mainLayout.addWidget(rightWidget)

        mainLayout.setStretch(0, 200)
        mainLayout.setStretch(1, 40)

        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)

        title = TitleLabel('Predictions')
        title.setStyleSheet('color: ' + firstColorName)
        title.setMinimumSize(100, 50)
        title.setBaseSize(100, 50)
        title.setMaximumHeight(70)
        # title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # backButton = ImageViewer(QPixmap('backButton.jpeg'))

        # backButton = ImageViewer(QPixmap('Back.png'))
        backButton = QLabel('Back')
        backButton.mousePressEvent = self.pressedBack
        # backButton.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # backButton = QLabel('Hello')

        topWidget = QWidget()
        topWidgetLayout = QGridLayout()
        backButton.setStyleSheet('padding: 0px;' +
                                 'margins: 0px;' +
                                 'border: 0px;')
        topWidgetLayout.addWidget(backButton, 0, 0, 0, 0, alignment=Qt.AlignLeft)
        topWidgetLayout.addWidget(title, 0, 0, 1, 2, alignment=Qt.AlignHCenter)

        # topWidgetLayout.setStretch(0,0)
        # topWidgetLayout.setStretch(1,99)
        topWidget.setLayout(topWidgetLayout)

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(topWidget)
        windowLayout.addWidget(mainWidget)
        windowLayout.setStretch(0,1)
        windowLayout.setStretch(1, 10)
        centralWidget = QWidget()
        centralWidget.setLayout(windowLayout)

        # self.setGeometry(300,100,800, 400)
        #self.setCentralWidget(centralWidget)
        thisWidgetsLayout = QHBoxLayout()
        thisWidgetsLayout.addWidget(centralWidget)
        thisWidgetsLayout.setContentsMargins(0,0,0,0)
        self.setLayout(thisWidgetsLayout)
        #self.show()

    # def resizeEvent(self, event):
    #     print(self.size())
    #     return super(QMainWindow, self).resizeEvent(event)

    def getVideos(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        # dlg.setFilter("Text files (*.txt)")
        dlg.exec_()
        filenames = dlg.selectedFiles()

        for path in filenames:
            # TODO: stop the user from being able to put in the wrong files
            name = path.split('/')[-1]
            scrollLabel = ScrollLabel(name)
            scrollLabel.path = path
            scrollLabel.mousePressEvent = \
                functools.partial(self.pressedScrollLabel, path=scrollLabel.path)
            self.vboxForScrollArea.addWidget(scrollLabel)


        # if len(filenames):
        #     # self.label.setPixmap(QPixmap(filenames[0]))
        #     path = filenames[0]
        #     name = filenames[0].split('/')[-1]
        #     scrollLabel = ScrollLabel(name)
        #     scrollLabel.path = path
        #     scrollLabel.mousePressEvent = \
        #         functools.partial(self.pressedScrollLabel, path=scrollLabel.path )
        #     self.vboxForScrollArea.addWidget(scrollLabel)

            # self.verticalWidgetForScrollArea.setLayout(self.vboxForScrollArea)
            # self.scroll1.setWidget(self.verticalWidgetForScrollArea)
            # self.scroll1.show()
            # self.scroll1.update()
            # self.update()
            # print('after scroll1')
    def getModels(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        # dlg.setFilter("Text files (*.txt)")
        dlg.exec_()
        filenames = dlg.selectedFiles()
        if len(filenames):
            splitText = filenames[0].split('/')
            self.modelLabel.setText(splitText[-1])
            self.modelPath = filenames[0]

    def getGrid(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.exec_()
        filenames = dlg.selectedFiles()
        if len(filenames):
            splitText = filenames[0].split('/')
            self.gridLabel.setText(splitText[-1])
            self.gridPath = filenames[0]

    def pressed(self,*arg, **kwargs):
        print('You pressed me')
        return

    def pressedScrollLabel(self, event, path=None):
        # self.label.setPixmap(QPixmap( path ))
        self.widget.initializePlayer(path)
        self.widget.ready = True
        # self.widget.play()

    def pressedGridLabel(self, event):
        # if len(self.drawingItems) > 0:
        #     for item in self.drawingItems:
        #         self.widget._scene.removeItem(item)
        #     self.drawingItems = []
        #     return

        print('You pressed the grid label')
        grid = np.load(self.gridPath)
        self.widget.setGrid(grid)
        return
        newWidth = self.widget.newWidth
        grid *= newWidth
        for circle in grid:
            x, y, r = circle
            diameter = 2 * r

            ellipse_item = QtWidgets.QGraphicsEllipseItem((x) - (diameter / 2),
                                                                (y) - (diameter / 2), diameter, diameter)
            ellipse_item.setPen(QtGui.QPen(QtCore.Qt.red))
            self.drawingItems.append(ellipse_item)
            self.widget._scene.addItem(ellipse_item)

    def pressedBack(self, event):
        self.parent().parent().backPressed()
        print('You pressed back')

    def run(self):
        # Real version
        # if self.gridPath is None or self.modelPath is None:
        #     print('You did not load a grid or model')
        #     return
        # grid = np.load(self.gridPath)
        # amount = self.vboxForScrollArea.count()
        # videoPaths = []
        # for labelIdx in range(amount):
        #     path = self.vboxForScrollArea.itemAt(labelIdx).widget().path
        #     videoPaths.append(path)

        # Fast testing version
        self.gridPath = 'grids/wellplate.npy'
        grid = np.load(self.gridPath)
        self.modelPath = 'models/resnet_pose_best_python_230608_four_blocks.pt'
        videoPaths = ['videos/wellPlateImages']

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
        grid[:,2] = 49
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
        bgsubList = bgsubFolder(folderPath)
        frame0 = bgsubList[0]
        # self.predictForFrame(frame0, grid)
        self.predictForFrames(bgsubList[:10], grid)
        print('You predicted for the folder')

    def predictForFrames(self, images, grid):
        green = [0, 255, 0]
        red = [0, 0, 255]
        rgb = np.stack((images[0], images[0], images[0]), axis=2)
        cutOutList = []
        circIdx = 0
        amountOfCircles = grid.shape[0]
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
            pose_recon = model_int8(im)

            # pose_recon = pose_recon.detach().cpu().numpy()
            # im = np.squeeze(im.detach().cpu().numpy())

            # # The following is extra computation stuff lets take it out for now
            # pose_recon = pose_recon.detach().cpu().numpy()
            # im = np.squeeze(im.cpu().detach().numpy())
            #
            # for imIdx in range(im.shape[0]):
            #     im1 = im[imIdx, ...]
            #     im1 *= 255
            #     im1 = im1.astype(np.uint8)
            #     pt1 = pose_recon[imIdx, ...]
            #
            #     noFishThreshold = 10
            #     if np.max(pt1) < noFishThreshold:
            #         # This is just a placeholder
            #         jj = 5
            #
            #     else:
            #
            #         # pt1 = pt1.astype(int)
            #         # im1[pt1[1,:], pt1[0,:]] =  255
            #         # cv.imwrite('test.png', im1)
            #         # exit()
            #
            #         # Fix this part up, should try to get rid of using np.where
            #         nonZero = np.where(im1 > 0)
            #         sY = np.min(nonZero[0])
            #         sX = np.min(nonZero[1])
            #         pt1[0, :] -= sX
            #         pt1[1, :] -= sY
            #
            #         circ = grid[circIdx]
            #         center = (int(circ[0]), int(circ[1]))
            #         radius = int(circ[2])
            #
            #         sX = center[0] - radius
            #         bX = center[0] + radius
            #         sY = center[1] - radius
            #         bY = center[1] + radius
            #         # sX, sY, bX, bY = boxes[ imIdx, ...]
            #         pt1[0, :] += sX
            #         pt1[1, :] += sY
            #         pt1 = pt1.astype(int)
            #
            #         rgb[pt1[1, :10], pt1[0, :10]] = green
            #         rgb[pt1[1, 10:], pt1[0, 10:]] = red
            #
            #     circIdx += 1
            #     if circIdx == amountOfCircles: circIdx = 0



        end = time.time()
        print("Finished predicting")
        print('duration: ', end - start)
    def predictForFrame(self, image, grid):
        green = [0, 255, 0]
        red = [0, 0, 255]
        rgb = np.stack((image, image, image), axis = 2)
        cutOutList = []
        circIdx = 0
        amountOfCircles = grid.shape[0]
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

        for cutoutIdx, cutout in enumerate(cutOutList):
            cv.imwrite('temp2/well_' + str(cutoutIdx) + '.png', cutout)
            # Prepping the data to give to resnet
        transform = transforms.Compose([padding(), transforms.PILToTensor()])
        data = CustomImageDataset(cutOutList, transform=transform)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.nworkers,
                            prefetch_factor=self.pftch_factor, persistent_workers=True)

        for i, im in enumerate(loader):
            im = im.to(self.device)
            pose_recon = self.resnetModel(im)

            # pose_recon = pose_recon.detach().cpu().numpy()
            # im = np.squeeze(im.detach().cpu().numpy())

            pose_recon = pose_recon.detach().cpu().numpy()
            im = np.squeeze(im.cpu().detach().numpy())

            for imIdx in range(im.shape[0]):
                im1 = im[imIdx, ...]
                im1 *= 255
                im1 = im1.astype(np.uint8)
                pt1 = pose_recon[imIdx, ...]

                noFishThreshold = 10
                if np.max(pt1) < noFishThreshold:
                    # This is just a placeholder
                    jj = 5

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
                    pt1 = pt1.astype(int)

                    rgb[pt1[1, :10], pt1[0, :10]] = green
                    rgb[pt1[1, 10:], pt1[0, 10:]] = red

                circIdx += 1
                if circIdx == amountOfCircles: circIdx = 0


        print('done predicting')
        cv.imwrite('temp.png', rgb)
if __name__ == '__main__':
    from Testing import *
    TestingWindow.testingClass = PredictionPage
    run()



