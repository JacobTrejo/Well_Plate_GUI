from GUI_Pages.Auxilary import *
import os
import time
import cv2 as cv
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets
from PyQt5.QtMultimediaWidgets import QVideoWidget
import numpy as np
from PyQt5.QtCore import QRect


def unNormalizeGrid(grid, offsets, scaledSize):
    xOffset, yOffset = offsets
    xSize, ySize = scaledSize

    # # The old version
    # gridArray = np.array(grid)
    # gridArray[:, 0] *= xSize
    # gridArray[:, 1] *= ySize
    # gridArray[:, 0] += xOffset
    # gridArray[:, 1] += yOffset

    # The new version
    gridArray = np.array(grid)
    gridArray[:, 0] *= xSize
    gridArray[:, 1] *= xSize
    gridArray[:, 0] += xOffset
    gridArray[:, 1] += yOffset

    gridArray[:, 2] *= (xSize)
    # gridArray[:, 2] *= (ySize **2 + xSize **2)**.5

    gridArray = np.round(gridArray).astype(int)

    return list(gridArray)

class ImageViewer(QLabel):
    pixmap = None
    _sizeHint = QSize()
    ratio = Qt.KeepAspectRatio
    transformation = Qt.SmoothTransformation

    def __init__(self, pixmap=None):
        super().__init__()
        self.setPixmap(pixmap)
        self.grid = None

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

        if self.grid is not None:
            qp.setPen(Qt.magenta)
            qp.setBrush(QBrush(QColor(0, 0, 0, 0)))

            imHeight, imWidth = self.scaled.height(), self.scaled.width()

            x_offset = (self.width() - imWidth) / 2
            y_offset = (self.height() - imHeight) / 2
            grid = unNormalizeGrid(self.grid, (x_offset, y_offset), (imWidth, imHeight))
            for circle in grid:
                x, y, r = circle
                # print(x, y, r)
                # x, y, r = int(round(x)), int(round(y)), int(round(r))
                center = QtCore.QPoint(x, y)
                qp.drawEllipse(center, r, r)


class FolderVideoPlayer(QWidget):

    def __init__(self, *args, **kwargs):
        super(FolderVideoPlayer, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        myLayout = QVBoxLayout()

        self.imageViewer = ImageViewer()
        self.folderName = None

        # # Temporary
        # self.folderName = 'videos/wellPlateImages'
        # self.imageList = os.listdir(self.folderName)
        # self.imageList.sort()
        # amount = len(self.imageList)
        # path0 = self.imageList[0]
        # self.imageViewer.setPixmap(QPixmap(self.folderName + '/' + path0))


        myLayout.addWidget(self.imageViewer, 1)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.valueChange)
        myLayout.addWidget(self.slider)

        # self.slider.setRange(0, amount - 1)

        self.setLayout(myLayout)

    def valueChange(self):
        self.imageViewer.setPixmap(QPixmap(self.folderName + '/' + self.imageList[int(self.slider.value())]))

    def setFolderName(self, folderName):
        self.folderName = folderName
        self.imageList = os.listdir(self.folderName)
        self.imageList.sort()
        amount = len(self.imageList)
        path0 = self.imageList[0]
        self.imageViewer.setPixmap(QPixmap(self.folderName + '/' + path0))
        self.slider.setRange(0, amount - 1)

    def setGrid(self, grid):
        self.imageViewer.grid = grid
        self.imageViewer.update()

    def removeGrid(self):
        self.imageViewer.grid = None
        self.imageViewer.update()

    def removeVideoFolder(self):
        self.folderName = None
        self.imageViewer.setPixmap(None)
        self.imageViewer.updateScaled()

# from Testing import *
# TestingWindow.testingClass = FolderVideoPlayer
# run()


