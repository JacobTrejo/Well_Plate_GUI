import sys

from pathlib import Path

import random
import imageio
import functools
import time
import numpy as np
import cv2 as cv

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QBasicTimer, QDate, QMimeData
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QSize
from PyQt5.Qt import QPainter
from PyQt5 import QtWidgets



im = imageio.imread('wellplate.png')

# theme colors
blue = '#435585'
whiteBlue = '#818FB4'
white = '#F5E8C7'

buttonStyleSheet = \
            '''
            border: 4px solid ''' + whiteBlue + ''';
            color: ''' + white + ''';
            font-family: 'shanti';
            font-size: 16px;
            border-radius: 25px;
            padding: 15px 0;
            margin-top: 20px}
            *:hover{
                background:  ''' + whiteBlue + '''
            }
            '''
smallerButtonStyleSheet = \
'''
            border: 2px solid ''' + whiteBlue + ''';
            color: ''' + white + ''';
            font-family: 'shanti';
            border-radius: 15px;
            padding: 2px 0;
            margin-top: 2px}
            *:hover{
                background:  ''' + whiteBlue + '''
            }
            '''
class MyQPixmap(QPixmap):

    def __init__(self, path):
        super(MyQPixmap, self).__init__(path)
        self.arr = imageio.imread(path)
        self.arrH, self.arrW = self.arr.shape[:2]

class FishLabel(QLabel):

    counter = 0

    def __init__(self, *args, **kwargs):
        super(FishLabel, self).__init__(*args, **kwargs)
        self.idx = FishLabel.counter
        self.cutout = None
        FishLabel.counter += 1

class ImageWidget(QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)

    def hasHeightForWidth(self):
        #return self.pixmap() is not None
        return True

    def heightForWidth(self, w):
        if self.pixmap():
            return int(w * (self.pixmap().height() / self.pixmap().width()))

    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.Resize :
            self.setPixmap(self.pixmap().scaled(
                self.width(), self.height(),
                QtCore.Qt.KeepAspectRatio))
            return True
        return True


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


class ClickableImageViewer(QLabel):
    pixmap = None
    _sizeHint = QSize()
    ratio = Qt.KeepAspectRatio
    transformation = Qt.SmoothTransformation

    def __init__(self, pixmap=None):
        super().__init__()
        self.setPixmap(pixmap)

        self.points = []
        self.amountOfPoints = 0

        self.cutoutWidget = None
        self.cutouts = []

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

        if self.points:
            imHeight, imWidth = self.scaled.height(), self.scaled.width()

            x_offset = (self.width() - imWidth) / 2
            y_offset = (self.height() - imHeight) / 2
            print('drawing points')
            # pen = QPen(Qt.red)
            # pen.setWidth(10)
            qp.setPen( QPen(QColor(0,0,0,0)) )
            qp.setBrush(Qt.red)
            for point in self.points:
                # qp.drawPoint(int(point[0]), int( point[1]) )
                qp.drawEllipse(int((point[0] * imWidth) + x_offset ), int( (point[1] * imHeight ) + y_offset ), 5,5)

            if self.amountOfPoints == 2:
                pointsArray = np.array(self.points)
                pointsArray[:, 0] = (pointsArray[:,0] * imWidth) + x_offset
                pointsArray[:, 1] = (pointsArray[:,1] * imHeight) + y_offset
                pointsArray = pointsArray.astype(int)

                qp.setBrush(QBrush(QColor(0,0,0,0)))
                qp.setPen(QPen(QColor(0,0,0,255)))
                qp.drawRect(QtCore.QRect(
                    QtCore.QPoint(pointsArray[0,0], pointsArray[0,1]),
                    QtCore.QPoint(pointsArray[1,0], pointsArray[1,1]) ))

    def mousePressEvent(self, ev):

        # converting to position in the pixmap
        imHeight, imWidth = self.scaled.height(), self.scaled.width()


        x_offset = (self.width() - imWidth ) /2
        y_offset = (self.height() - imHeight ) /2
        y, x = ev.pos().y() - y_offset, ev.pos().x() - x_offset
        print('x: ', x)
        print('y: ', y)
        # We have to check if the point is in bounds
        if x >= 0 and x <= imWidth and y >=0  and y <= imHeight:
            self.amountOfPoints = (self.amountOfPoints + 1) % 3

            if self.amountOfPoints == 0:
                self.points = []
            else:
                self.points.append([x / imWidth, y / imHeight])

            self.update()
            if self.amountOfPoints == 2:
                cv.imwrite('temp.png', self.getCutout())
                self.cutoutWidget.setPixmap(QPixmap('temp.png'))
                # self.cutoutWidget = ImageViewer(QPixmap('temp.png'))
                # self.cutoutWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                # self.cutoutWidget.setStyleSheet('border: 2px solid black;')
                # self.cutoutWidget.adjustSize()
                # self.frameLayout.setRowStretch(1,1)
            else:
                self.cutoutWidget.setPixmap(None)


    def getCutout(self):
        pointsArray = np.array(self.points)
        h, w = self.pixmap.arrH, self.pixmap.arrW
        pointsArray[:,0] *= w
        pointsArray[:,1] *= h
        pointsArray = pointsArray.astype(int)

        sx, bx = min(pointsArray[:,0]), max(pointsArray[:,0])
        sy, by = min(pointsArray[:,1]), max(pointsArray[:,1])

        return self.pixmap.arr[sy: by + 1,sx: bx + 1]

class DrawingWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(DrawingWindow, self).__init__(*args, **kwargs)
        self.setMinimumSize(854, 510)
        self.amountOfFishSaved = 0
        self.vidArray = np.load('vid.npy')
        self.frameIdx = 0
        self.amountOfFrames = self.vidArray.shape[-1]
        self.initUI()

    def initUI(self):

        # Lets create the left widget this will compose of the video display and buttons for frame selection
        # self.setStyleSheet('border: 1 px')
        # Creating the bottom bar
        bottomBar = QFrame()
        # bottomBar.setStyleSheet('QWidget {border: 1px solid;}')
        bottomBarLayout = QVBoxLayout()
        bottomBarLayout.setSpacing(0)
        bottomBarLayout.setContentsMargins(0,0,0,0)
        # Creating the frame
        frameSelectionBar = QFrame()
        # frameSelectionBar.setStyleSheet('border: 1px solid')
        frameSelectionBarLayout = QVBoxLayout()
        frameSelectionBarLayout.setSpacing(0)
        frameSelectionBarLayout.setContentsMargins(0, 0, 0, 0)
        frameSelectionBarWidgets = QFrame()
        frameSelectionBarWidgetsLayout = QHBoxLayout()
        frameSelectionBarWidgetsLayout.setContentsMargins(0,0,0,0)
        frameSelectionBarWidgetsLayout.setSpacing(0)
        # Now the actual elements
        backButton = QPushButton('Back')
        backButton.clicked.connect(self.previousFrame)
        self.frameSelectionLabel = QLineEdit('1')
        self.frameSelectionLabel.returnPressed.connect(self.pressedReturn)
        self.frameSelectionLabel.setStyleSheet('margin: 0px')
        self.frameSelectionLabel.setContentsMargins(5,0,5,0)
        self.frameSelectionLabel.setMaximumWidth(100)
        forwardButton = QPushButton('Next')
        forwardButton.clicked.connect(self.nextFrame)
        # Adding them
        frameSelectionBarWidgetsLayout.addWidget(backButton)
        frameSelectionBarWidgetsLayout.addWidget(self.frameSelectionLabel, alignment=Qt.AlignHCenter)
        frameSelectionBarWidgetsLayout.addWidget(forwardButton)
        frameSelectionBarWidgets.setLayout(frameSelectionBarWidgetsLayout)
        frameSelectionBarLayout.addWidget(frameSelectionBarWidgets, alignment=Qt.AlignHCenter)
        frameSelectionBar.setLayout(frameSelectionBarLayout)
        # Adding this to the bottom bar
        bottomBarLayout.addWidget(frameSelectionBar)
        self.onFrameLabel = QLabel('On Frame 1 of ' + str(self.amountOfFrames))
        bottomBarLayout.addWidget(self.onFrameLabel, alignment=Qt.AlignHCenter)
        bottomBar.setLayout(bottomBarLayout)


        self.label = ClickableImageViewer(MyQPixmap('wellplate.png'))

        leftWidget = QWidget()
        leftWidgetLayout = QVBoxLayout()
        leftWidgetLayout.addWidget(self.label,1)
        leftWidgetLayout.addWidget(bottomBar, 0)
        leftWidget.setLayout(leftWidgetLayout)
        # leftWidget.setStyleSheet('border: 1px solid black')
        leftWidgetLayout.setContentsMargins(0,0,0,0)
        leftWidgetLayout.setSpacing(20)
        print( leftWidgetLayout.getContentsMargins() )
        # Preparing the main widget, the display and the side bar

        sideBar = QWidget()
        # sideBar.setStyleSheet('padding: 0px;' +
        #                       'margin: 0px;' +
        #                       'border: 1px solid black;')
        sideBarLayout = QVBoxLayout()

        frame1 = QFrame()
        frame1Layout = QGridLayout()
        frame1Layout.addWidget(QLabel('Cutout'), 0,0,alignment=Qt.AlignHCenter)
        # self.cutOutWidget = ImageViewer(QPixmap('Back.png'))
        self.cutOutWidget = ImageViewer()
        self.cutOutWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.cutoutWidget = self.cutOutWidget
        frame1Layout.addWidget(self.cutOutWidget,1,0)
        frame1Layout.setRowStretch(0,0)
        frame1Layout.setRowStretch(1,1 )
        frame1.setLayout(frame1Layout)
        # frame1.setStyleSheet('border: 1px solid')

        frame2 = QFrame()
        frame2Layout = QVBoxLayout()
        saveCutoutButton = QPushButton('Save Cutout')
        saveCutoutButton.clicked.connect( self.pressedSaveFish )
        self.cutoutsSavedLabel = QLabel('Amount Saved: 0')
        frame2Layout.addWidget(saveCutoutButton, 1, alignment=Qt.AlignCenter)
        frame2Layout.addWidget(self.cutoutsSavedLabel, 1, alignment=Qt.AlignCenter)
        frame2.setLayout(frame2Layout)

        frame3 = QFrame()
        frame3.setStyleSheet('border: 1px solid')
        frame3Layout = QGridLayout()
        cutoutsSavedLabel = QLabel('Cutouts Saved')
        cutoutsSavedLabel.setStyleSheet('border: 0px')
        frame3Layout.addWidget(cutoutsSavedLabel,0,0,alignment=Qt.AlignHCenter )
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.widgetForScrollArea = QWidget()
        self.vBoxForScrollArea = QVBoxLayout()
        self.vBoxForScrollArea.setContentsMargins(0,0,0,20)
        self.widgetForScrollArea.setLayout(self.vBoxForScrollArea)
        self.widgetForScrollArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scrollArea.setWidget(self.widgetForScrollArea)

        # frame3Layout.addWidget(scrollArea,1,0,alignment=Qt.AlignCenter)
        frame3Layout.addWidget(self.scrollArea,1,0)

        frame3Layout.setRowStretch(0,0)
        frame3Layout.setRowStretch(1,1)
        frame3.setLayout(frame3Layout)

        frame4 = QFrame()
        frame4Layout = QVBoxLayout()
        startAnnotatingButton = QPushButton('Start Annotating')
        startAnnotatingButton.clicked.connect(self.startAnnotating)
        changeVideoButton = QPushButton('Change Video')
        changeVideoButton.clicked.connect( self.getNewVideo )

        frame4Layout.addWidget(startAnnotatingButton, 1, alignment=Qt.AlignCenter)
        frame4Layout.addWidget(changeVideoButton, 1, alignment=Qt.AlignCenter)
        frame4.setLayout(frame4Layout)

        sideBarLayout.addWidget(frame1,1)
        sideBarLayout.addWidget(frame2, 1)
        sideBarLayout.addWidget(frame3,1)
        sideBarLayout.addWidget(frame4,1)

        sideBar.setLayout(sideBarLayout)

        mainWidget = QWidget()
        mainWidgetLayout = QHBoxLayout()
        # mainWidgetLayout.addWidget(self.label, 200)
        mainWidgetLayout.addWidget(leftWidget, 130)
        mainWidgetLayout.addWidget(sideBar, 40)
        mainWidget.setLayout(mainWidgetLayout)

        # The main widget was created now lets just create the title
        topWidget = QWidget()
        topWidgetLayout = QGridLayout()

        backButton = ImageViewer(QPixmap('Back.png'))
        title = QLabel('Intrinsic Parameters')
        topWidgetLayout.addWidget(backButton, 0,0,0,0, alignment=Qt.AlignLeft)
        topWidgetLayout.addWidget(title, 0, 0, 1, 0, alignment=Qt.AlignHCenter )
        topWidget.setLayout(topWidgetLayout)

        # Now let's add the topWidget and the mainWidget
        centralWidget = QWidget()
        centralWidgetLayout = QVBoxLayout()
        centralWidgetLayout.addWidget(topWidget,1)
        centralWidgetLayout.addWidget(mainWidget, 10)
        centralWidget.setLayout(centralWidgetLayout)

        self.setCentralWidget(centralWidget)

        # self.setGeometry(300,300, 800, 450)
        self.show()

    def getNewVideo(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.exec_()
        filenames = dlg.selectedFiles()
        if len(filenames):
            self.vidArray = np.load(filenames[0])
            self.frameIdx = 0
            self.amountOfFrames = self.vidArray.shape[-1]
            self.changeFrame()
            self.cutOutWidget.setPixmap(None)
            self.label.amountOfPoints = 0
            self.label.points = []
    def pressedSaveFish(self):
        if self.label.amountOfPoints == 2:
            self.amountOfFishSaved += 1
            cutout = self.label.getCutout()
            fishLabel = FishLabel('Fish ' + str(self.amountOfFishSaved))
            fishLabel.cutout = cutout
            print(fishLabel.idx)
            fishLabel.mousePressEvent = \
                functools.partial(self.pressedSavedLabel,fishLabel)
            self.vBoxForScrollArea.addWidget(fishLabel)
            self.update()
            self.cutoutsSavedLabel.setText('Amount Saved: ' + str(self.amountOfFishSaved))

    def pressedSavedLabel(self, fishLabel, event):
        if event.button() == Qt.RightButton:
            print('right button pressed')

            # self.vBoxForScrollArea.removeWidget(fishLabel)
            # fishLabel.close()
            # self.cutOutWidget.setPixmap(None)

            menu = QMenu()
            menu.addAction('Remove', functools.partial( self.removeFishLabel, fishLabel) )
            menu.exec_(QCursor().pos())


            return
        cv.imwrite('temp.png', fishLabel.cutout)
        self.cutOutWidget.setPixmap(QPixmap('temp.png'))
        self.cutOutWidget.update()

    def removeFishLabel(self, fishLabel):

        self.vBoxForScrollArea.removeWidget(fishLabel)
        fishLabel.close()
        del fishLabel
        self.cutOutWidget.setPixmap(None)

        self.amountOfFishSaved += -1
        self.cutoutsSavedLabel.setText('Amount Saved: ' + str(self.amountOfFishSaved))

    def pressedReturn(self):
        desiredFrame = self.frameSelectionLabel.text()
        if desiredFrame.isnumeric():
            desiredFrame = int(desiredFrame)
            self.frameIdx = np.clip(desiredFrame - 1, 0, self.amountOfFrames)
            self.changeFrame()

    def nextFrame(self):
        self.frameIdx = (self.frameIdx + 1) % self.amountOfFrames
        self.changeFrame()

    def previousFrame(self):
        self.frameIdx = (self.frameIdx - 1) % self.amountOfFrames
        self.changeFrame()

    def changeFrame(self):
        self.onFrameLabel.setText('On Frame ' + str(self.frameIdx + 1) + ' of ' + str(self.amountOfFrames))
        self.frameSelectionLabel.setText(str(self.frameIdx + 1))
        frame = self.vidArray[..., self.frameIdx]
        cv.imwrite('temp.png', frame)
        self.label.amountOfPoints = 0
        self.label.points = []
        self.label.setPixmap(MyQPixmap('temp.png'))

    def startAnnotating(self):
        # print('amount of labels: ', self.vBoxForScrollArea.count())
        cutouts = []
        for idx in range(self.vBoxForScrollArea.count()):
            cutout = self.vBoxForScrollArea.itemAt(idx).widget().cutout
            cutouts.append(cutout)
        dialogWindow = AnnotationsDialog(cutouts)
        dialogWindow.exec_()

############ Testing the QDialog Window
class ClickableCutoutViewer(QLabel):
    pixmap = None
    _sizeHint = QSize()
    ratio = Qt.KeepAspectRatio
    transformation = Qt.SmoothTransformation

    def __init__(self, pixmap=None):
        super().__init__()
        self.setPixmap(pixmap)

        self.points = []
        self.amountOfPoints = 0


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

        if self.points:
            imHeight, imWidth = self.scaled.height(), self.scaled.width()

            x_offset = (self.width() - imWidth) / 2
            y_offset = (self.height() - imHeight) / 2
            print('drawing points')
            # pen = QPen(Qt.red)
            # pen.setWidth(10)
            qp.setPen( QPen(QColor(0,0,0,0)) )
            qp.setBrush(Qt.red)
            for point in self.points:
                # qp.drawPoint(int(point[0]), int( point[1]) )
                qp.drawEllipse(int((point[0] * imWidth) + x_offset ), int( (point[1] * imHeight ) + y_offset ), 5,5)

            # if self.amountOfPoints == 2:
            #     pointsArray = np.array(self.points)
            #     pointsArray[:, 0] = (pointsArray[:,0] * imWidth) + x_offset
            #     pointsArray[:, 1] = (pointsArray[:,1] * imHeight) + y_offset
            #     pointsArray = pointsArray.astype(int)
            #
            #     qp.setBrush(QBrush(QColor(0,0,0,0)))
            #     qp.setPen(QPen(QColor(0,0,0,255)))
            #     qp.drawRect(QtCore.QRect(
            #         QtCore.QPoint(pointsArray[0,0], pointsArray[0,1]),
            #         QtCore.QPoint(pointsArray[1,0], pointsArray[1,1]) ))

    def mousePressEvent(self, ev):

        # converting to position in the pixmap
        imHeight, imWidth = self.scaled.height(), self.scaled.width()


        x_offset = (self.width() - imWidth ) /2
        y_offset = (self.height() - imHeight ) /2
        y, x = ev.pos().y() - y_offset, ev.pos().x() - x_offset
        print('x: ', x)
        print('y: ', y)
        # We have to check if the point is in bounds
        if x >= 0 and x <= imWidth and y >=0  and y <= imHeight:
            self.amountOfPoints = (self.amountOfPoints + 1) % 3

            if self.amountOfPoints == 0:
                self.points = []
            else:
                self.points.append([x / imWidth, y / imHeight])

            self.update()
            # if self.amountOfPoints == 2:
            #     cv.imwrite('temp.png', self.getCutout())
            #     self.cutoutWidget.setPixmap(QPixmap('temp.png'))
            #     # self.cutoutWidget = ImageViewer(QPixmap('temp.png'))
            #     # self.cutoutWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            #     # self.cutoutWidget.setStyleSheet('border: 2px solid black;')
            #     # self.cutoutWidget.adjustSize()
            #     # self.frameLayout.setRowStretch(1,1)
            # else:
            #     self.cutoutWidget.setPixmap(None)




# class QDialogTester(QMainWindow):
#
#     def __init__(self, *args, **kwargs):
#         super(QDialogTester, self).__init__(*args, **kwargs)
#         self.initUI()
#
#     def initUI(self):
#         self.setGeometry(300,300, 450, 450)
#         self.show()
#
#         dialogWindow = QDialog(self)
#         dialogWindow.setGeometry(300,300,550,400)
#         dialogWindow.setWindowTitle('Annotating Window')
#
#         # centralWidget = QWidget()
#         centralWidgetLayout = QVBoxLayout()
#
#         title = QLabel('Fish Annotations')
#         centralWidgetLayout.addWidget(title, 0,alignment=Qt.AlignHCenter)
#
#         # centralWidget.setLayout(centralWidgetLayout)
#
#         # Creating the main window, will contain the display widget on the left and the side bar
#         mainWidget = QWidget()
#         mainWidgetLayout = QHBoxLayout()
#         mainWidgetLayout.setContentsMargins(0,0,0,0)
#         #mainWidget.setStyleSheet('border: 1px solid')
#         # Creating the left widget
#         leftWidget = QWidget()
#         leftWidgetLayout = QVBoxLayout()
#         leftWidgetLayout.setContentsMargins(0,0,0,0)
#
#         cutoutViewer = ClickableCutoutViewer()
#         leftWidgetLayout.addWidget(cutoutViewer,1)
#
#         #   Creating the bottom bar
#         bottomBar = QWidget()
#         bottomBarLayout = QVBoxLayout()
#         bottomBarLayout.setContentsMargins(0,0,0,0)
#         #   Creating the buttons to iterate through the fish
#         buttonsBar = QWidget()
#         buttonsBarLayout = QHBoxLayout()
#         buttonsBarLayout.setContentsMargins(0,0,0,0)
#         backButton = QPushButton('Back')
#         nextButton = QPushButton('Next')
#         buttonsBarLayout.addWidget(backButton)
#         buttonsBarLayout.addWidget(nextButton)
#         buttonsBar.setLayout(buttonsBarLayout)
#         bottomBarLayout.addWidget(buttonsBar)
#         bottomBar.setLayout(bottomBarLayout)
#
#         # TODO: replace this with the amount of labels
#         onCutoutLabel = QLabel('On Cutout 1 of ' + str(1))
#         bottomBarLayout.addWidget(onCutoutLabel, alignment=Qt.AlignHCenter)
#
#         leftWidgetLayout.addWidget(bottomBar)
#         leftWidget.setLayout(leftWidgetLayout)
#
#         mainWidgetLayout.addWidget(leftWidget, 2)
#
#         # Creating the right window
#         rightWidget = QWidget()
#         rightWidgetLayout = QVBoxLayout()
#         rightWidgetLayout.setContentsMargins(0,0,0,0)
#         # rightWidget.setStyleSheet('border: 1px solid')
#
#         #   Creating the top section of the right widget
#         rightWidgetTopSection = QWidget()
#         rightWidgetTopSectionLayout = QVBoxLayout()
#         rightWidgetTopSectionLayout.setContentsMargins(0,0,0,0)
#
#         scrollAreaTitle = QLabel('Cutouts')
#         rightWidgetTopSectionLayout.addWidget(scrollAreaTitle,0, alignment=Qt.AlignHCenter)
#
#         scrollArea = QScrollArea()
#         scrollArea.setWidgetResizable(True)
#         scrollArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         scrollAreaWidget = QWidget()
#         scrollAreaWidgetLayout = QVBoxLayout()
#
#         rightWidgetTopSectionLayout.addWidget(scrollArea,1)
#         rightWidgetTopSection.setLayout(rightWidgetTopSectionLayout)
#         rightWidgetLayout.addWidget(rightWidgetTopSection, 2)
#
#         rightWidgetBottomSection = QWidget()
#         rightWidgetBottomSectionLayout = QVBoxLayout()
#         rightWidgetBottomSectionLayout.setContentsMargins(0,0,0,0)
#
#         rightWidgetBottomTopSection = QWidget()
#         rightWidgetBottomTopSectionLayout = QVBoxLayout()
#         rightWidgetBottomTopSectionLayout.setContentsMargins(0,0,0,0)
#
#         saveAnnotationButton = QPushButton('Mark Annotation')
#         amountSavedLabel = QLabel('Annotations Saved: 0 of ' + str(90))
#         rightWidgetBottomTopSectionLayout.addWidget(amountSavedLabel, 1)
#         rightWidgetBottomTopSectionLayout.addWidget(saveAnnotationButton, 1)
#         rightWidgetBottomTopSection.setLayout(rightWidgetBottomTopSectionLayout)
#
#         doneButton = QPushButton('Done')
#
#         rightWidgetBottomSectionLayout.addWidget(rightWidgetBottomTopSection, 1)
#         rightWidgetBottomSectionLayout.addWidget(doneButton, 1)
#
#         rightWidgetBottomSection.setLayout(rightWidgetBottomSectionLayout)
#
#         #rightWidgetLayout.addWidget(scrollArea,2)
#         rightWidgetLayout.addWidget(rightWidgetBottomSection,1)
#
#         #rightWidgetBottomSectionLayout.addWidget(saveAnnotationButton)
#         #rightWidgetBottomSectionLayout.addWidget(doneButton)
#         #rightWidgetBottomSection.setLayout(rightWidgetBottomSectionLayout)
#
#         #rightWidgetLayout.addWidget(rightWidgetBottomSection, 1)
#         rightWidget.setLayout(rightWidgetLayout)
#
#         # rightWidget.setStyleSheet('border: 1px solid')
#
#         mainWidgetLayout.addWidget(rightWidget, 1)
#
#
#         mainWidget.setLayout(mainWidgetLayout)
#
#         # mainWidget.setStyleSheet('border: 1px solid')
#
#         centralWidgetLayout.addWidget(mainWidget, 1)
#
#         dialogWindow.setLayout(centralWidgetLayout)
#
#
#
#
#
#
#
#         dialogWindow.exec_()


class AnnotationsDialog(QDialog):

    def __init__(self, cutouts,*args, **kwargs):
        super(AnnotationsDialog, self).__init__(*args, **kwargs)
        self.cutouts = cutouts
        self.initUI()
    def initUI(self):
        self.setGeometry(300,300,550,400)
        self.setWindowTitle('Annotating Window')

        # centralWidget = QWidget()
        centralWidgetLayout = QVBoxLayout()

        title = QLabel('Fish Annotations')
        centralWidgetLayout.addWidget(title, 0,alignment=Qt.AlignHCenter)

        # centralWidget.setLayout(centralWidgetLayout)

        # Creating the main window, will contain the display widget on the left and the side bar
        mainWidget = QWidget()
        mainWidgetLayout = QHBoxLayout()
        mainWidgetLayout.setContentsMargins(0,0,0,0)
        #mainWidget.setStyleSheet('border: 1px solid')
        # Creating the left widget
        leftWidget = QWidget()
        leftWidgetLayout = QVBoxLayout()
        leftWidgetLayout.setContentsMargins(0,0,0,0)

        cutout0 = self.cutouts[0]
        cv.imwrite('temp.png', cutout0)
        cutoutViewer = ClickableCutoutViewer(QPixmap('temp.png'))
        leftWidgetLayout.addWidget(cutoutViewer,1)

        #   Creating the bottom bar
        bottomBar = QWidget()
        bottomBarLayout = QVBoxLayout()
        bottomBarLayout.setContentsMargins(0,0,0,0)
        #   Creating the buttons to iterate through the fish
        buttonsBar = QWidget()
        buttonsBarLayout = QHBoxLayout()
        buttonsBarLayout.setContentsMargins(0,0,0,0)
        backButton = QPushButton('Back')
        nextButton = QPushButton('Next')
        buttonsBarLayout.addWidget(backButton)
        buttonsBarLayout.addWidget(nextButton)
        buttonsBar.setLayout(buttonsBarLayout)
        bottomBarLayout.addWidget(buttonsBar)
        bottomBar.setLayout(bottomBarLayout)

        # TODO: replace this with the amount of labels
        onCutoutLabel = QLabel('On Cutout 1 of ' + str(1))
        bottomBarLayout.addWidget(onCutoutLabel, alignment=Qt.AlignHCenter)

        leftWidgetLayout.addWidget(bottomBar)
        leftWidget.setLayout(leftWidgetLayout)

        mainWidgetLayout.addWidget(leftWidget, 2)

        # Creating the right window
        rightWidget = QWidget()
        rightWidgetLayout = QVBoxLayout()
        rightWidgetLayout.setContentsMargins(0,0,0,0)
        # rightWidget.setStyleSheet('border: 1px solid')

        #   Creating the top section of the right widget
        rightWidgetTopSection = QWidget()
        rightWidgetTopSectionLayout = QVBoxLayout()
        rightWidgetTopSectionLayout.setContentsMargins(0,0,0,0)

        scrollAreaTitle = QLabel('Cutouts')
        rightWidgetTopSectionLayout.addWidget(scrollAreaTitle,0, alignment=Qt.AlignHCenter)

        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scrollAreaWidget = QWidget()
        scrollAreaWidgetLayout = QVBoxLayout()

        rightWidgetTopSectionLayout.addWidget(scrollArea,1)
        rightWidgetTopSection.setLayout(rightWidgetTopSectionLayout)
        rightWidgetLayout.addWidget(rightWidgetTopSection, 2)

        rightWidgetBottomSection = QWidget()
        rightWidgetBottomSectionLayout = QVBoxLayout()
        rightWidgetBottomSectionLayout.setContentsMargins(0,0,0,0)

        rightWidgetBottomTopSection = QWidget()
        rightWidgetBottomTopSectionLayout = QVBoxLayout()
        rightWidgetBottomTopSectionLayout.setContentsMargins(0,0,0,0)

        saveAnnotationButton = QPushButton('Mark Annotation')
        amountSavedLabel = QLabel('Annotations Saved: 0 of ' + str(90))
        rightWidgetBottomTopSectionLayout.addWidget(amountSavedLabel, 1, alignment=Qt.AlignHCenter)
        rightWidgetBottomTopSectionLayout.addWidget(saveAnnotationButton, 1)
        rightWidgetBottomTopSection.setLayout(rightWidgetBottomTopSectionLayout)

        doneButton = QPushButton('Done')

        rightWidgetBottomSectionLayout.addWidget(rightWidgetBottomTopSection, 1)
        rightWidgetBottomSectionLayout.addWidget(doneButton, 1)

        rightWidgetBottomSection.setLayout(rightWidgetBottomSectionLayout)

        #rightWidgetLayout.addWidget(scrollArea,2)
        rightWidgetLayout.addWidget(rightWidgetBottomSection,1)

        #rightWidgetBottomSectionLayout.addWidget(saveAnnotationButton)
        #rightWidgetBottomSectionLayout.addWidget(doneButton)
        #rightWidgetBottomSection.setLayout(rightWidgetBottomSectionLayout)

        #rightWidgetLayout.addWidget(rightWidgetBottomSection, 1)
        rightWidget.setLayout(rightWidgetLayout)

        # rightWidget.setStyleSheet('border: 1px solid')

        mainWidgetLayout.addWidget(rightWidget, 1)


        mainWidget.setLayout(mainWidgetLayout)

        # mainWidget.setStyleSheet('border: 1px solid')

        centralWidgetLayout.addWidget(mainWidget, 1)

        self.setLayout(centralWidgetLayout)







        # dialogWindow.exec_()

class QDialogTester2(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(QDialogTester2, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        self.setGeometry(300,300, 450, 450)
        self.show()

        dialogWindow = AnnotationsDialog()
        dialogWindow.exec_()


app = QApplication(sys.argv)
ex = DrawingWindow()
#ex = QDialogTester2()
# ex = predictionWindow()
# ex = Window()

sys.exit(app.exec())











