from Auxilary import *
import os
import time
import cv2 as cv

from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets
from PyQt5.QtMultimediaWidgets import QVideoWidget
import numpy as np
from PyQt5.QtCore import QRect
def calculateMaximumSize(ogSize, frameSize):
    width , height = ogSize
    maxWidth, maxHeight = frameSize

    heightFromMaxWidth = maxWidth * ( height / width)

    factor = 1

    if heightFromMaxWidth < maxHeight:
        newHeight = heightFromMaxWidth
        return ( factor * np.floor(maxWidth), factor * np.floor(newHeight))
        # return ((maxWidth), (newHeight))

    else:
        newWidth = maxHeight * (width / height)
        return ( factor *  np.floor(newWidth) , factor * np.floor(maxHeight) )
        # return ((newWidth), (maxHeight))



class Widget(QtWidgets.QWidget):

    def __init__(self, parent=None):

        super(Widget, self).__init__(parent)

        # This variable holds the grid
        self.grid = None

        self._scene = QtWidgets.QGraphicsScene(self)


        self._gv = QtWidgets.QGraphicsView(self._scene)
        # self._gv = QtWidgets.QGraphicsView()

        self._gv.setContentsMargins(0,0,0,0)
        self.setMinimumWidth(0)
        self.setMinimumHeight(0)

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


        self._ellipse_item = QtWidgets.QGraphicsEllipseItem(QtCore.QRectF(0, 0, 40, 40), self._videoitem)
        # self._ellipse_item.setBrush(QtGui.QBrush(QtCore.Qt.black))
        self._ellipse_item.setPen(QtGui.QPen(QtCore.Qt.red))
        self._scene.addItem(self._ellipse_item)


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
        myLayout.setContentsMargins(0,0,0,0)
        myLayout.setSpacing(0)
        # myLayout.addWidget(self._gv, 0,0,1,1)
        myLayout.addWidget(self._gv, 1)
        self.setLayout(myLayout)
        # self.setStyleSheet('border: 1px solid white')

        # self.show()

        self.initializePlayer()
        self.play()
        
    def play(self):
        self._player.play()

    def initializePlayer(self):
        self._player = QtMultimedia.QMediaPlayer(self, QtMultimedia.QMediaPlayer.VideoSurface)
        self._player.setVideoOutput(self._videoitem)
        # self._player.setVideoOutput(videoWidget)
        #file = os.path.join(os.path.dirname(__file__), "video.mp4")#video.mp4 is under the same dirctory
        self._player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile('/Users/jacobtrejo/PycharmProjects/Well_Plate_GUI_2/videosActual/zebrafish.mp4')))

        # Getting its aspect ratio
        vid = cv.VideoCapture('/Users/jacobtrejo/PycharmProjects/Well_Plate_GUI_2/videosActual/zebrafish.mp4')
        self.height = vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.width = vid.get(cv.CAP_PROP_FRAME_WIDTH)





    def resizeEvent(self, a0):
        # size = QtCore.QSizeF(a0.size().width(), a0.size().height())
        # size = QtCore.QSizeF(800, 500)

        newWidth, newHeight = calculateMaximumSize((self.width, self.height), (a0.size().width(), a0.size().height()))
        self._videoitem.setSize( QtCore.QSizeF(newWidth, newHeight  ) )
        self._gv.resize(int(newWidth), int(newHeight))
        self._gv.setMinimumWidth(int(0))
        self._gv.setMinimumHeight( int(0))
        # self._gv.fitInView(self._videoitem, Qt.KeepAspectRatio)

        #self._scene.setSceneRect(QtCore.QRectF(0,0,a0.size().width() - 5, a0.size().height() - 5))
        # self._scene.setSceneRect(QtCore.QRectF(0, -30, newWidth, newHeight))
        sceneRectOffset = (a0.size().height() ) / 2

        self._scene.setSceneRect(QtCore.QRectF(0, 0, a0.size().width(), a0.size().height()))

        self._scene.removeItem(self._ellipse_item)
        diameter = 40
        self._ellipse_item = QtWidgets.QGraphicsEllipseItem((.5 * newWidth) - (diameter /2), (.5 * newHeight) - (diameter / 2), diameter, diameter)
        self._scene.addItem(self._ellipse_item)


        print('widget height: ', a0.size().height())
        print('scene height: ', newHeight)

        # print(self._scene.height())

        # print(self._scene.minimumRenderSize())

        # self._scene.setSize( QtCore.QSizeF(newWidth, newHeight  ) )



        # self._gv.resize(int(newWidth), int(newHeight))
        # print(self._gv.size())
        # print(self._scene.__sizeof__())
        # print(a0.size())
        # super(Widget, self).resizeEvent(a0)



        # newHeight = a0.size().width() * (self.height / self.width)

        # self._videoitem.setSize( QtCore.QSizeF(a0.size().width(), a0.size().height()  ) )


        # print('the scene,', self._gv.size())
        # self._videoitem.setSize(a0.size())


from Testing import *
TestingWindow.testingClass = Widget
run()


# class TestingWindow(QMainWindow):
#
#     def __init__(self, *args, **kwargs):
#         super(TestingWindow, self).__init__(*args, **kwargs)
#         self.initUI()
#
#     def initUI(self):
#         centralWidget = QWidget()
#         centralWidgetLayout = QHBoxLayout()
#
#         leftSpace = QWidget()
#         videoView = Widget()
#         rightSpace = QWidget()
#
#         centralWidgetLayout.addWidget(leftSpace, 1)
#
#         centralCentralWidget = QWidget()
#         centralCentralWidgetLayout = QVBoxLayout()
#         leftSpace2 = QWidget()
#         rightSpace2 = QWidget()
#
#         centralCentralWidgetLayout.addWidget(leftSpace2, 1)
#         centralCentralWidgetLayout.addWidget(videoView, 1)
#         centralCentralWidgetLayout.addWidget(rightSpace2, 1)
#
#         centralCentralWidget.setLayout(centralCentralWidgetLayout)
#         centralWidgetLayout.addWidget(centralCentralWidget, 1)
#
#         # centralWidgetLayout.addWidget(videoView, 1)
#         centralWidgetLayout.addWidget(rightSpace, 1)
#
#         centralWidget.setLayout(centralWidgetLayout)
#
#         self.setCentralWidget(centralWidget)
#         self.setGeometry(300, 300, 700, 300)
#         self.show()
#
#
# if __name__ == '__main__':
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     w = TestingWindow()
#
#     w.show()
#     sys.exit(app.exec_())