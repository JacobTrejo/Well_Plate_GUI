from GUI_Pages.Auxilary import *
import os
import time
import cv2 as cv
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets
from PyQt5.QtMultimediaWidgets import QVideoWidget
import numpy as np
from PyQt5.QtCore import QRect


def calculateMaximumSize(ogSize, frameSize):
    width, height = ogSize
    maxWidth, maxHeight = frameSize

    heightFromMaxWidth = maxWidth * (height / width)

    factor = 1

    if heightFromMaxWidth < maxHeight:
        newHeight = heightFromMaxWidth
        return (factor * np.floor(maxWidth), factor * np.floor(newHeight))
        # return ((maxWidth), (newHeight))

    else:
        newWidth = maxHeight * (width / height)
        return (factor * np.floor(newWidth), factor * np.floor(maxHeight))
        # return ((newWidth), (maxHeight))


class Widget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)

        # The following variables have to be passed in
        self.playButton = None
        self.positionSlider = None

        # This variable holds the grid
        self.grid = None
        self.drawingItems = []
        self.ready = False

        self._scene = QtWidgets.QGraphicsScene(self)

        self._gv = QtWidgets.QGraphicsView(self._scene)
        # self._gv = QtWidgets.QGraphicsView()

        self._gv.setContentsMargins(0, 0, 0, 0)
        self.setMinimumWidth(0)
        self.setMinimumHeight(0)

        self._gv.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._gv.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # gvLayout = QHBoxLayout()
        # leftStretcher = QWidget()
        # gvLayout.addWidget(leftStretcher, 1)

        # construct a videoitem for showing the video
        self._videoitem = QtMultimediaWidgets.QGraphicsVideoItem()

        # self._videoitem.setSize(QtCore.QSizeF(800, 500))

        # videoWidget = QVideoWidget()

        self._scene.addItem(self._videoitem)
        # self._scene.addWidget(self._videoitem)

        # self._scene.addItem(videoWidget)
        # self._scene.addItem(self._gv)

        # Some other original lines
        # self._ellipse_item = QtWidgets.QGraphicsEllipseItem(QtCore.QRectF(0, 0, 40, 40), self._videoitem)
        # # self._ellipse_item.setBrush(QtGui.QBrush(QtCore.Qt.black))
        # self._ellipse_item.setPen(QtGui.QPen(QtCore.Qt.red))
        # self._scene.addItem(self._ellipse_item)
        self._ellipse_item = None

        # self._gv.fitInView(self._videoitem, Qt.KeepAspectRatioByExpanding)

        # self._gv.fitInView(videoWidget)

        # The Original Lines
        # self._player = QtMultimedia.QMediaPlayer(self, QtMultimedia.QMediaPlayer.VideoSurface)
        # self._player.setVideoOutput(self._videoitem)
        # # self._player.setVideoOutput(videoWidget)
        # #file = os.path.join(os.path.dirname(__file__), "video.mp4")#video.mp4 is under the same dirctory
        # self._player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile('/Users/jacobtrejo/PycharmProjects/Well_Plate_GUI_2/videosActual/zebrafish.mp4')))
        #
        # # Getting its aspect ratio
        # vid = cv.VideoCapture('/Users/jacobtrejo/PycharmProjects/Well_Plate_GUI_2/videosActual/zebrafish.mp4')
        # self.height = vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        # self.width = vid.get(cv.CAP_PROP_FRAME_WIDTH)

        # self._player.play()

        # size = QtCore.QSizeF(500, 300)  # I hope it can fullscreen the video
        # self._videoitem.setSize(size)

        # self._gv.showFullScreen()
        # self._gv.show()

        # self._player.play()

        myLayout = QHBoxLayout()
        myLayout.setContentsMargins(0, 0, 0, 0)
        myLayout.setSpacing(0)
        # myLayout.addWidget(self._gv, 0,0,1,1)
        myLayout.addWidget(self._gv, 1)
        self.setLayout(myLayout)
        # self.setStyleSheet('border: 1px solid white')

        # self.show()

        # self.initializePlayer('/Users/jacobtrejo/PycharmProjects/Well_Plate_GUI_2/videosActual/zebrafish.mp4')
        # self.play()

    def play(self):
        self._player.play()

    def toggledPlay(self):
        if not self.ready: return
        if self._player.state() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def connectSlider(self, slider):
        self.positionSlider = slider
        self.positionSlider.sliderMoved.connect(self.changePosition)

    def changePosition(self, position):
        self._player.setPosition(position)

    def initializePlayer(self, url):
        self._player = QtMultimedia.QMediaPlayer(self, QtMultimedia.QMediaPlayer.VideoSurface)
        self._player.setVideoOutput(self._videoitem)

        self._player.durationChanged.connect(self.durationChanged)
        self._player.positionChanged.connect(self.positionChanged)

        # self._player.setVideoOutput(videoWidget)
        # file = os.path.join(os.path.dirname(__file__), "video.mp4")#video.mp4 is under the same dirctory
        self._player.setMedia(QtMultimedia.QMediaContent(
            QtCore.QUrl.fromLocalFile(url)))

        # Getting its aspect ratio
        vid = cv.VideoCapture(url)
        self.height = vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.width = vid.get(cv.CAP_PROP_FRAME_WIDTH)

        # This was temp added
        newWidth, newHeight = calculateMaximumSize((self.width, self.height), (self.size().width(), self.size().height()))
        self.newWidth = newWidth

        self._videoitem.setSize(QtCore.QSizeF(newWidth, newHeight))
        self._gv.resize(int(newWidth), int(newHeight))
        self._gv.setMinimumWidth(int(0))
        self._gv.setMinimumHeight(int(0))
        # self._gv.fitInView(self._videoitem, Qt.KeepAspectRatio)

        # self._scene.setSceneRect(QtCore.QRectF(0,0,a0.size().width() - 5, a0.size().height() - 5))
        # self._scene.setSceneRect(QtCore.QRectF(0, -30, newWidth, newHeight))
        sceneRectOffset = (self.size().height()) / 2

        # # Drawing the grid
        # grid = np.load('grids/testingGrid.npy')
        # grid *= newWidth
        #
        # for circle in grid:
        #     x, y, r = circle
        #     diameter = 2 * r
        #     self._ellipse_item = QtWidgets.QGraphicsEllipseItem((x) - (diameter / 2),
        #                                                         (y) - (diameter / 2), diameter, diameter)
        #
        #     self._scene.addItem(self._ellipse_item)

        self._scene.setSceneRect(QtCore.QRectF(0, 0, self.size().width(), self.size().height()))
        self._player.play()
        self._player.pause()
        self._player.setPosition(0)
        self.positionSlider.setValue(0)
    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def positionChanged(self, position):
        # print('position changed was called')
        self.positionSlider.setValue(position)

    def resizeEvent(self, a0):
        if not self.ready: return
        # size = QtCore.QSizeF(a0.size().width(), a0.size().height())
        # size = QtCore.QSizeF(800, 500)

        newWidth, newHeight = calculateMaximumSize((self.width, self.height), (a0.size().width(), a0.size().height()))
        self.newWidth = newWidth
        self._videoitem.setSize(QtCore.QSizeF(newWidth, newHeight))
        self._gv.resize(int(newWidth), int(newHeight))
        self._gv.setMinimumWidth(int(0))
        self._gv.setMinimumHeight(int(0))
        # self._gv.fitInView(self._videoitem, Qt.KeepAspectRatio)

        # self._scene.setSceneRect(QtCore.QRectF(0,0,a0.size().width() - 5, a0.size().height() - 5))
        # self._scene.setSceneRect(QtCore.QRectF(0, -30, newWidth, newHeight))
        sceneRectOffset = (a0.size().height()) / 2

        self._scene.setSceneRect(QtCore.QRectF(0, 0, a0.size().width(), a0.size().height()))



        # The following are for drawing
        # self._scene.removeItem(self._ellipse_item)

        # diameter = 40
        # self._ellipse_item = QtWidgets.QGraphicsEllipseItem((.5 * newWidth) - (diameter / 2),
        #                                                     (.5 * newHeight) - (diameter / 2), diameter, diameter)
        #
        # self._scene.addItem(self._ellipse_item)

        print('widget height: ', a0.size().height())
        print('scene height: ', newHeight)
        # removing the older ones
        for item in self.drawingItems:
            self._scene.removeItem(item)
        self.drawingItems = []

        if self.grid is not None:
            grid = self.grid * self.newWidth
            for circle in grid:
                x, y, r = circle
                diameter = 2 * r

                ellipse_item = QtWidgets.QGraphicsEllipseItem((x) - (diameter / 2),
                                                              (y) - (diameter / 2), diameter, diameter)
                ellipse_item.setPen(QtGui.QPen(QtCore.Qt.red))
                self.drawingItems.append(ellipse_item)
                self._scene.addItem(ellipse_item)

    def setGrid(self, grid):
        self.grid = grid
        # self.update()

    def removeGrid(self):

        for item in self.drawingItems:
            self._scene.removeItem(item)
        self.drawingItems = []

        self.grid = None

    def removeVideo(self):
        self._scene.removeItem(self._videoitem)
        self._videoitem = QtMultimediaWidgets.QGraphicsVideoItem()

        # self._videoitem.setSize(QtCore.QSizeF(800, 500))

        # videoWidget = QVideoWidget()

        self._scene.addItem(self._videoitem)




