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

from PyQt5.QtCore import QSize
from PyQt5.Qt import QPainter

blue = '#435585'
whiteBlue = '#818FB4'
white = '#F5E8C7'


firstColorName = '#012F34'
secondColorName = '#0E484E'
thirdColorName = '#417C81'
fourthColorName = '#97B9BE'

secondColorName = '#1F5E67'
# thirdColorName = '#7DAAB0'
# fourthColorName = '#CEE2E5'
# blue = thirdColorName
# whiteBlue = secondColorName
# white = fourthColorName

blue = secondColorName
whiteBlue = thirdColorName
white = fourthColorName
# white = firstColorName


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


class MenuPage(QWidget):

    def __init__(self, *args, **kwargs):
        super(MenuPage, self).__init__(*args, **kwargs)

        self.initUI()

    def initUI(self):
        self.setStyleSheet('background: ' + blue + ';')
        buttonsStyleSheet = \
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

        grid = QGridLayout(self)

        class titleLabel(QLabel):
            def __init__(self, *args, **kwargs):
                super(titleLabel, self).__init__(*args, **kwargs)
                self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

            def resizeEvent(self, a0):
                font = self.font()
                font.setPixelSize(int(self.height() * .7))
                self.setFont(font)
                # self.setAlignment(Qt.AlignmentFlag.AlignCenter)
                # self.adjustSize()
                # self.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        title = titleLabel('Title')

        title = QLabel('Title')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setMaximumHeight(100)
        # title.setBaseSize(title.width(), 100)
        title.setMinimumSize(100, 50)
        grid.addWidget(title, 0, 0)

        button = QPushButton('Predict')
        button.setStyleSheet(buttonsStyleSheet)
        button.setFixedWidth(300)
        button.clicked.connect(self.predictionPressed)
        grid.addWidget(button, 1, 0)

        ccBtn = QPushButton('Calculate Correlation Coefficients')
        ccBtn.setFixedWidth(300)
        ccBtn.setStyleSheet(buttonsStyleSheet)
        grid.addWidget(ccBtn, 2, 0)

        wellsBtn = QPushButton('Define Wells')
        wellsBtn.setFixedWidth(300)
        wellsBtn.setStyleSheet(buttonsStyleSheet)
        wellsBtn.clicked.connect(self.defineWellsPressed)
        grid.addWidget(wellsBtn, 3, 0)

        parmsBtn = QPushButton('Calculate Intrinsic Parameters')
        parmsBtn.setFixedWidth(300)
        parmsBtn.clicked.connect(self.intrinsicParametersPressed)
        parmsBtn.setStyleSheet(buttonsStyleSheet)
        grid.addWidget(parmsBtn, 4, 0)

        temp = QWidget()
        temp.setLayout(grid)
        tempLayout = QVBoxLayout()
        tempLayout.setContentsMargins(0,0,0,0)
        tempLayout.addWidget(temp)

        self.setLayout(tempLayout)

    def predictionPressed(self):
        self.parent().parent().predictionsPressed()
        # print('You pressed the predictions button')

    def defineWellsPressed(self):
        self.parent().parent().defineWellsPressed()

    def intrinsicParametersPressed(self):
        self.parent().parent().intrinsicParametersPressed()

# class PredictionsPage(QWidget):
#
#     def __init__(self, *args, **kwargs):
#         super(PredictionsPage, self).__init__(*args, **kwargs)
#         self.initUI()
#
#     def initUI(self):
#
#         'Define the UI Here'

#   For Predictions

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

class PredictionWindow(QWidget):

    def __init__(self, *args, **kwargs):
        super(PredictionWindow, self).__init__(*args, **kwargs)

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
        self.gridLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        # adding the elements to the fourth frame
        layout4 = QHBoxLayout()
        layout4.addWidget(button4)
        layout4.setStretch(0,1)
        self.frame4.setLayout(layout4)
        self.frame4.setStyleSheet('border: 1px solid black')
        self.setMinimumSize(854, 510)
        self.initUI()

    def initUI(self):
        rightLayout = QVBoxLayout()
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
        self.label.setStyleSheet('border: 1px solid black;' +
                                 'color: white;')
        self.label.setText('No Video Selected')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.show()
        # self.label.adjustSize()
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.label)
        mainLayout.addWidget(rightWidget)

        mainLayout.setStretch(0, 200)
        mainLayout.setStretch(1, 40)

        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)

        title = TitleLabel('Predictions')
        title.setMinimumSize(100, 50)
        title.setBaseSize(100, 50)
        title.setMaximumHeight(70)
        # title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # backButton = ImageViewer(QPixmap('backButton.jpeg'))
        backButton = ImageViewer(QPixmap('Back.png'))
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

    def getGrid(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.exec_()
        filenames = dlg.selectedFiles()
        if len(filenames):
            splitText = filenames[0].split('/')
            self.gridLabel.setText(splitText[-1])

    def pressed(self,*arg, **kwargs):
        print('You pressed me')
        return

    def pressedScrollLabel(self, event, path=None):
        self.label.setPixmap(QPixmap( path ))

    def pressedBack(self, event):
        self.parent().parent().backPressed()
        print('You pressed back')


# The define wells program
def returnCircleParameters(point1, point2, point3):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3

    A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    B = (x1**2 + y1**2)*(y3 - y2) + (x2**2 + y2**2)*(y1 - y3) + (x3**2 + y3**2)*(y2 - y1)
    C = (x1**2 + y1**2)*(x2 - x3) + (x2**2 + y2**2)*(x3 - x1) + (x3**2 + y3**2)*(x1 - x2)
    D = (x1**2 + y1**2)*(x3*y2 - x2*y3) + (x2**2 + y2**2) * (x1*y3 - x3*y1) + (x3**2 + y3**2)*(x2*y1 - x1*y2)

    xc = (-1 * B)/ (2 * A)
    yc = (-1 * C)/ (2 * A)
    r = ((B**2 + C**2 - 4*A*D) / (4 * A**2)) ** .5

    return (xc, yc), r

def estimateGridFrom2Corners(circle1, circle2):
    (x1, y1), r1 = circle1
    (x2, y2), r2 = circle2

    sx, bx = min(x1,x2), max(x1,x2)
    sy, by = min(y1, y2), max(y1, y2)

    r = np.mean((r1, r2))

    dx = bx - sx
    dy = by - sy
    amountOfColumns = round(dx / (2*r)) + 1
    amountOfRows = round(dy / (2*r)) + 1

    stepX = dx / (amountOfColumns - 1)
    stepY = dy / (amountOfRows -1)

    grid = []
    for rowIdx in range(amountOfRows):
        for colIdx in range(amountOfColumns):
            grid.append([sx + colIdx * stepX, sy + rowIdx * stepY, r])

    return(grid)

    # print('columns: ', amountOfColumns)
    # print('rows: ', amountOfRows)

def normalizeCircle(circle, offsets, scaledSize):
    """ The single dimension case of normalizeGrid"""
    (x, y), r = circle
    xOffset, yOffset = offsets
    xSize, ySize = scaledSize

    x -= xOffset
    y -= yOffset
    x *= (1/xSize)
    y *= (1/xSize)
    r *= (1/xSize)

    return [x, y, r]

def normalizeGrid(grid, offsets, scaledSize):
    xOffset, yOffset = offsets
    xSize, ySize = scaledSize

    # # The old version
    # gridArray = np.array(grid)
    # gridArray[:, 0] -= xOffset
    # gridArray[:, 0] *= (1/ xSize)
    # gridArray[:, 1] -= yOffset
    # gridArray[:, 1] *= (1/ ySize)

    # # The new version
    gridArray = np.array(grid)
    gridArray[:, 0] -= xOffset
    gridArray[:, 0] *= (1 / xSize)
    gridArray[:, 1] -= yOffset
    gridArray[:, 1] *= (1 / xSize)

    # Since the aspect ratio is kept the same the dimension we use to normalize does not matter
    gridArray[:, 2] *= (1 / (xSize))
    # gridArray[:, 2] *= (1/ ((ySize**2 + xSize**2)**.5) )

    return list(gridArray)

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

def getSelectedObject(point, grid, offset, scaledSize):
    distanceThreshold = .01

    x, y = point
    xOffset, yOffset = offset
    xSize, ySize = scaledSize
    x, y = x - xOffset, y - yOffset
    x, y = x / xSize, y / xSize

    gridArray = np.array(grid)

    distanceFromCenters = ((gridArray[:,0] - x) ** 2 + (gridArray[:, 1] - y) ** 2) ** .5
    minDistanceFromCenterIdx = np.argmin(distanceFromCenters)
    minDistanceFromCenter = distanceFromCenters[minDistanceFromCenterIdx]

    distanceFromRadius = np.abs(gridArray[:, 2] - distanceFromCenters)
    minDistanceFromRadiusIdx = np.argmin(distanceFromRadius)
    minDistanceFromRadius = distanceFromRadius[minDistanceFromRadiusIdx]

    if minDistanceFromCenter < distanceThreshold or minDistanceFromRadius < distanceThreshold:
        if minDistanceFromCenter < minDistanceFromRadius:
            return (0, minDistanceFromCenterIdx)
        else:
            return (1, minDistanceFromRadiusIdx)
    else:
        return None

    # print('minimum distance from radius: ', distanceFromRadius[minDistanceFromRadiusIdx])


class GridEstimatorImageViewer(QLabel):
    pixmap = None
    _sizeHint = QSize()
    ratio = Qt.KeepAspectRatio
    transformation = Qt.SmoothTransformation


    def __init__(self, pixmap=None, *args, **kwargs):
        super(GridEstimatorImageViewer, self).__init__(*args, **kwargs)
        self.setMouseTracking(True)

        self.setPixmap(pixmap)

        self.points = []
        self.amountOfPoints = 0

        self.circle1 = None
        self.circle2 = None
        self.grid = None
        self.selectedObject = None

    def setPixmap(self, pixmap):
        if self.pixmap != pixmap:
            self.points = []
            self.amountOfPoints = 0

            self.grid = None
            self.selectedObject = None

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

        if self.grid:
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

            if self.selectedObject:
                qp.setPen(Qt.cyan)
                idx = self.selectedObject[1]
                x, y, r = grid[idx]
                center = QtCore.QPoint(x, y)
                if self.selectedObject[0]:
                    # Its the radius
                    qp.drawEllipse(center, r, r)
                else:
                    qp.setBrush(Qt.cyan)
                    qp.drawEllipse(center, 2, 2)


            return

        if self.points:
            imHeight, imWidth = self.scaled.height(), self.scaled.width()

            x_offset = (self.width() - imWidth) / 2
            y_offset = (self.height() - imHeight) / 2
            print('drawing points')
            # pen = QPen(Qt.red)
            # pen.setWidth(10)
            qp.setPen( QPen(QColor(0,0,0,0)) )
            qp.setBrush(Qt.red)


            if len(self.points) < 3:

                for point in self.points:
                    # qp.drawPoint(int(point[0]), int( point[1]) )
                    # qp.drawEllipse(int((point[0] * imWidth) + x_offset ), int( (point[1] * imHeight ) + y_offset ), 5,5)
                    center = QtCore.QPoint( int( round((point[0] * imWidth) + x_offset ) ), int( round((point[1] * imHeight ) + y_offset) ) )
                    qp.drawEllipse(center, 2,2)
            else:
                # We will use the first three points to draw a circle
                qp.setPen(Qt.magenta)
                qp.setBrush(QBrush(QColor(0, 0, 0, 0)))
                translatedPoints = np.array(self.points.copy())

                translatedPoints[:, 0] *= imWidth
                translatedPoints[:, 0] += x_offset
                translatedPoints[:, 1] *= imHeight
                translatedPoints[:, 1] += y_offset

                center, r = returnCircleParameters(translatedPoints[0], translatedPoints[1], translatedPoints[2])
                center0, r0 = center, r
                center = QtCore.QPoint(int(center[0]), int(center[1]))
                qp.drawEllipse(center, int(round(r)), int(round(r)))

                if len(self.points) == 6:
                    # Actually we should draw the grid

                    # We will draw the grid

                    # We will use the remaining three points to draw the second circle
                    center, r = returnCircleParameters(translatedPoints[3], translatedPoints[4], translatedPoints[5])

                    grid = estimateGridFrom2Corners((center0, r0), (center, r))

                    self.grid = normalizeGrid(grid, (x_offset, y_offset), (imWidth, imHeight))
                    # grid = unNormalizeGrid(self.grid, (x_offset, y_offset), (imWidth, imHeight))

                    for circle in grid:
                        x, y, r = circle
                        x, y, r = int(round(x)), int(round(y)), int(round(r))
                        center = QtCore.QPoint(x, y)
                        qp.drawEllipse(center, r, r)

                    # center = QtCore.QPoint(int(center[0]), int(center[1]))
                    # qp.drawEllipse(center, int(round(r)), int(round(r)))

                else:
                    qp.setPen(QPen(QColor(0, 0, 0, 0)))
                    qp.setBrush(Qt.red)
                    for point in self.points[3:]:
                        # qp.drawPoint(int(point[0]), int( point[1]) )
                        # qp.drawEllipse(int((point[0] * imWidth) + x_offset ), int( (point[1] * imHeight ) + y_offset ), 5,5)
                        center = QtCore.QPoint(int(round((point[0] * imWidth) + x_offset)),
                                               int(round((point[1] * imHeight) + y_offset)))
                        qp.drawEllipse(center, 2, 2)

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
        # if self.pixmap is None: return
        if self.selectedObject or self.pixmap is None: return
        # converting to position in the pixmap
        imHeight, imWidth = self.scaled.height(), self.scaled.width()


        x_offset = (self.width() - imWidth ) /2
        y_offset = (self.height() - imHeight ) /2
        y, x = ev.pos().y() - y_offset, ev.pos().x() - x_offset

        # We have to check if the point is in bounds
        if x >= 0 and x <= imWidth and y >=0  and y <= imHeight:
            self.amountOfPoints = (self.amountOfPoints + 1) % 7

            if self.amountOfPoints == 0:
                self.points = []
                self.grid = None
            else:
                self.points.append([x / imWidth, y / imHeight])

            self.update()

    def mouseMoveEvent(self, ev):
        if self.grid is None: return

        if ev.buttons() == Qt.LeftButton and self.selectedObject:
            imHeight, imWidth = self.scaled.height(), self.scaled.width()

            x_offset = (self.width() - imWidth) / 2
            y_offset = (self.height() - imHeight) / 2

            x, y = ev.x(), ev.y()
            x, y = (x - x_offset)/ imWidth, (y - y_offset)/ imWidth
            if self.selectedObject[0]:
                x0, y0, r0 = self.grid[self.selectedObject[1]]
                r = ((x - x0)**2 + (y - y0)**2)**.5
                self.grid[self.selectedObject[1]] = x0, y0, r
            else:
                self.grid[self.selectedObject[1]] = x, y, self.grid[self.selectedObject[1]][-1]
            self.update()
            return

        imHeight, imWidth = self.scaled.height(), self.scaled.width()

        x_offset = (self.width() - imWidth) / 2
        y_offset = (self.height() - imHeight) / 2

        selectedObject = getSelectedObject((ev.x(), ev.y()), self.grid, (x_offset, y_offset), (imWidth, imHeight) )

        if selectedObject:
            self.selectedObject = selectedObject
            self.update()
        else:
            if self.selectedObject:
                self.selectedObject = None
                self.update()
            # self.selectedObject = None

        #print('X: ', ev.x())
        #print('Y: ', ev.y())

class IndividualWellImageViewer(QLabel):
    pixmap = None
    _sizeHint = QSize()
    ratio = Qt.KeepAspectRatio
    transformation = Qt.SmoothTransformation


    def __init__(self, pixmap=None):
        super().__init__()
        self.setMouseTracking(True)
        self.setPixmap(pixmap)

        self.points = []
        self.amountOfPoints = -1

        self.circle1 = None
        self.circle2 = None
        self.grid = []
        self.selectedObject = None

    def setPixmap(self, pixmap):
        if self.pixmap != pixmap:

            self.points = []
            self.amountOfPoints = -1
            self.grid = []
            self.selectedObject = None

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

        if self.grid:
            qp.setPen(Qt.magenta)
            qp.setBrush(QBrush(QColor(0, 0, 0, 0)))

            imHeight, imWidth = self.scaled.height(), self.scaled.width()

            x_offset = (self.width() - imWidth) / 2
            y_offset = (self.height() - imHeight) / 2

            # if len(self.grid) == 1:
            #     grid = unNormalizeGrid([self.grid], (x_offset, y_offset), (imWidth, imHeight))
            # else:
            #     grid = unNormalizeGrid(self.grid, (x_offset, y_offset), (imWidth, imHeight))

            grid = unNormalizeGrid(self.grid, (x_offset, y_offset), (imWidth, imHeight))

            for circle in grid:
                x, y, r = circle
                # print(x, y, r)
                # x, y, r = int(round(x)), int(round(y)), int(round(r))
                center = QtCore.QPoint(x, y)
                qp.drawEllipse(center, r, r)

            if self.selectedObject:
                qp.setPen(Qt.cyan)
                idx = self.selectedObject[1]
                x, y, r = grid[idx]
                center = QtCore.QPoint(x, y)
                if self.selectedObject[0]:
                    # Its the radius
                    qp.drawEllipse(center, r, r)
                else:
                    qp.setBrush(Qt.cyan)
                    qp.drawEllipse(center, 2, 2)


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
                # qp.drawEllipse(int((point[0] * imWidth) + x_offset ), int( (point[1] * imHeight ) + y_offset ), 5,5)
                center = QtCore.QPoint(int(round((point[0] * imWidth) + x_offset)),
                                       int(round((point[1] * imHeight) + y_offset)))
                qp.drawEllipse(center, 2, 2)
            return

            #
            # if len(self.points) < 3:
            #
            #     for point in self.points:
            #         # qp.drawPoint(int(point[0]), int( point[1]) )
            #         # qp.drawEllipse(int((point[0] * imWidth) + x_offset ), int( (point[1] * imHeight ) + y_offset ), 5,5)
            #         center = QtCore.QPoint( int( round((point[0] * imWidth) + x_offset ) ), int( round((point[1] * imHeight ) + y_offset) ) )
            #         qp.drawEllipse(center, 2,2)
            # else:
            #     # We will use the first three points to draw a circle
            #     qp.setPen(Qt.magenta)
            #     qp.setBrush(QBrush(QColor(0, 0, 0, 0)))
            #     translatedPoints = np.array(self.points.copy())
            #
            #     translatedPoints[:, 0] *= imWidth
            #     translatedPoints[:, 0] += x_offset
            #     translatedPoints[:, 1] *= imHeight
            #     translatedPoints[:, 1] += y_offset
            #
            #     center, r = returnCircleParameters(translatedPoints[0], translatedPoints[1], translatedPoints[2])
            #     center0, r0 = center, r
            #     center = QtCore.QPoint(int(center[0]), int(center[1]))
            #     qp.drawEllipse(center, int(round(r)), int(round(r)))
            #
            #     if len(self.points) == 6:
            #         # Actually we should draw the grid
            #
            #         # We will draw the grid
            #
            #         # We will use the remaining three points to draw the second circle
            #         center, r = returnCircleParameters(translatedPoints[3], translatedPoints[4], translatedPoints[5])
            #
            #         grid = estimateGridFrom2Corners((center0, r0), (center, r))
            #
            #         self.grid = normalizeGrid(grid, (x_offset, y_offset), (imWidth, imHeight))
            #         # grid = unNormalizeGrid(self.grid, (x_offset, y_offset), (imWidth, imHeight))
            #
            #         for circle in grid:
            #             x, y, r = circle
            #             x, y, r = int(round(x)), int(round(y)), int(round(r))
            #             center = QtCore.QPoint(x, y)
            #             qp.drawEllipse(center, r, r)
            #
            #         # center = QtCore.QPoint(int(center[0]), int(center[1]))
            #         # qp.drawEllipse(center, int(round(r)), int(round(r)))
            #
            #     else:
            #         qp.setPen(QPen(QColor(0, 0, 0, 0)))
            #         qp.setBrush(Qt.red)
            #         for point in self.points[3:]:
            #             # qp.drawPoint(int(point[0]), int( point[1]) )
            #             # qp.drawEllipse(int((point[0] * imWidth) + x_offset ), int( (point[1] * imHeight ) + y_offset ), 5,5)
            #             center = QtCore.QPoint(int(round((point[0] * imWidth) + x_offset)),
            #                                    int(round((point[1] * imHeight) + y_offset)))
            #             qp.drawEllipse(center, 2, 2)

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
        if self.pixmap is None: return
        if self.selectedObject and ev.buttons() == Qt.RightButton:
            menu = QMenu()
            # menu.addAction('Remove', functools.partial(self.removeG, fishLabel))
            # menu.addAction('Remove', self.removeCircle, self.selectedObject[1])
            menu.addAction('Remove', functools.partial(self.removeCircle, self.selectedObject[1]) )
            menu.exec_(QCursor().pos())
            return

        if self.selectedObject: return
        # converting to position in the pixmap
        imHeight, imWidth = self.scaled.height(), self.scaled.width()

        x_offset = (self.width() - imWidth ) /2
        y_offset = (self.height() - imHeight ) /2
        y, x = ev.pos().y() - y_offset, ev.pos().x() - x_offset

        # We have to check if the point is in bounds
        if x >= 0 and x <= imWidth and y >=0  and y <= imHeight:
            self.points.append([x / imWidth, y / imHeight])

            self.amountOfPoints = (self.amountOfPoints + 1) % 3

            if self.amountOfPoints == 2:
                translatedPoints = np.array(self.points)
                translatedPoints[:, 0] *= imWidth
                translatedPoints[:, 0] += x_offset
                translatedPoints[:, 1] *= imHeight
                translatedPoints[:, 1] += y_offset

                translatedPoints = translatedPoints.astype(float)

                circle = returnCircleParameters(translatedPoints[0], translatedPoints[1], translatedPoints[2])
                # circle = [center[0], center[1], radius]
                normalizedCircle = normalizeCircle(circle, (x_offset, y_offset), (imWidth, imHeight))
                self.grid.append(normalizedCircle)

                # self.grid.append(circle)
                # self.grid = normalizeGrid(self.grid, (x_offset, y_offset), (imWidth, imHeight))

                print(np.array(self.grid))
                self.points = []

                self.update()
                # exit()
                # self.grid = None
            # else:
            #     self.points.append([x / imWidth, y / imHeight])

            self.update()

    def mouseMoveEvent(self, ev):
        if len(self.grid) == 0: return

        if ev.buttons() == Qt.LeftButton and self.selectedObject:
            imHeight, imWidth = self.scaled.height(), self.scaled.width()

            x_offset = (self.width() - imWidth) / 2
            y_offset = (self.height() - imHeight) / 2

            x, y = ev.x(), ev.y()
            x, y = (x - x_offset)/ imWidth, (y - y_offset)/ imWidth
            if self.selectedObject[0]:
                x0, y0, r0 = self.grid[self.selectedObject[1]]
                r = ((x - x0)**2 + (y - y0)**2)**.5
                self.grid[self.selectedObject[1]] = x0, y0, r
            else:
                self.grid[self.selectedObject[1]] = x, y, self.grid[self.selectedObject[1]][-1]
            self.update()
            return

        imHeight, imWidth = self.scaled.height(), self.scaled.width()

        x_offset = (self.width() - imWidth) / 2
        y_offset = (self.height() - imHeight) / 2

        selectedObject = getSelectedObject((ev.x(), ev.y()), self.grid, (x_offset, y_offset), (imWidth, imHeight) )

        if selectedObject:
            self.selectedObject = selectedObject
            self.update()
        else:
            if self.selectedObject:
                self.selectedObject = None
                self.update()
            # self.selectedObject = None

    def removeCircle(self, index):
        self.grid.pop(index)
        self.selectedObject = None
        self.update()
        #print('X: ', ev.x())
        #print('Y: ', ev.y())


class DefineWindowClass(QWidget):

    def __init__(self, *args, **kwargs):
        super(DefineWindowClass, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        self.setStyleSheet('background: ' + blue + ';')
        # self.setGeometry(300,300,600,400)

        # gridEstimatorImageViewer = GridEstimatorImageViewer(QPixmap('wellplate.png'))

        self.centralWidget = QWidget(self)
        # centralWidget.setStyleSheet('border: 1px solid')
        centralWidgetLayout = QVBoxLayout()
        # centralWidgetLayout.setContentsMargins(0,0,0,0)
        # Creating the top bar
        topBar = QWidget()
        topBarLayout = QGridLayout()
        topBarLayout.setContentsMargins(0,0,0,0)
        backButton = QLabel("back")
        backButton.mousePressEvent = self.pressedBack

        title = QLabel("Define Wells")
        topBarLayout.addWidget(backButton, 0,0,0,0,alignment=Qt.AlignLeft)
        topBarLayout.addWidget(title, 0,0,1, 0, alignment=Qt.AlignHCenter)
        topBar.setLayout(topBarLayout)

        # Creating the main widget
        self.mainWidget = QWidget(self.centralWidget)
        self.mainWidgetLayout = QHBoxLayout()
        self.mainWidgetLayout.setContentsMargins(0,0,0,0)
        # mainWidgetLayout.setSpacing(0)

        #   Creating the left side, the imageViewer
        self.stack = QStackedWidget()
        # self.gridEstimatorImageViewer = GridEstimatorImageViewer(QPixmap('wellplate.png'))
        # self.individualImageViewer = IndividualWellImageViewer(QPixmap('wellplate.png'))
        self.gridEstimatorImageViewer = GridEstimatorImageViewer(None)
        self.individualImageViewer = IndividualWellImageViewer(None)

        self.gridEstimatorImageViewer.setText('No Image Selected')
        self.individualImageViewer.setText('No Image Selected')

        self.stack.addWidget(self.gridEstimatorImageViewer)
        self.stack.addWidget(self.individualImageViewer)
        #   Creating the right side, the sidebar
        sideBar = QWidget()
        # sideBar.setStyleSheet('border: 1px solid')
        sideBarLayout = QVBoxLayout()
        # sideBarLayout.setContentsMargins(0,0,0,0)

        scrollArea = QScrollArea()

        topFrame = QFrame()
        # topFrame = QTextEdit()
        # topFrame.setReadOnly(True)
        # topFrame.setMinimumSize(0,0)
        # topFrame.setStyleSheet('border: 1px solid')


        middleFrame = QWidget()
        # middleFrame.setStyleSheet('border: 1px solid')
        middleFrameLayout = QVBoxLayout()
        middleFrameLayout.setContentsMargins(10,0,10,0)
        radioButtons = QWidget()
        radioButtonsLayout = QHBoxLayout()
        radioButtonsLayout.setContentsMargins(0,0,0,0)

        estimateButton = QRadioButton('Estimate')
        estimateButton.setChecked(True)
        estimateButton.mode = 'ESTIMATE'
        estimateButton.toggled.connect(self.onToggle)
        radioButtonsLayout.addWidget(estimateButton)

        individualButton = QRadioButton('Individual')
        individualButton.mode = 'INDIVIDUAL'
        individualButton.toggled.connect(self.onToggle)
        radioButtonsLayout.addWidget(individualButton)
        radioButtons.setLayout(radioButtonsLayout)
        middleFrameLayout.addWidget(radioButtons)
        middleFrame.setLayout(middleFrameLayout)

        bottomFrame = QWidget()
        bottomFrameLayout = QVBoxLayout()
        saveGridButton = QPushButton('Save Grid')
        saveGridButton.clicked.connect(self.saveGrid)
        saveGridButton.setStyleSheet('')
        changeVidButton = QPushButton('Change Image')
        changeVidButton.clicked.connect( self.changedImage )
        bottomFrameLayout.addWidget(changeVidButton)
        bottomFrameLayout.addWidget(saveGridButton)
        bottomFrame.setLayout(bottomFrameLayout)

        # bottomFrame.setStyleSheet('border: 1px solid')

        sideBarLayout.addWidget(topFrame)
        sideBarLayout.addWidget(middleFrame)
        sideBarLayout.addWidget(bottomFrame)

        sideBar.setLayout(sideBarLayout)

        # adding to main widget

        # self.mainWidgetLayout.addWidget(self.gridEstimatorImageViewer, 5)
        self.mainWidgetLayout.addWidget(self.stack, 5, alignment = Qt.AlignCenter)
        self.mainWidgetLayout.addWidget(sideBar, 2)
        self.mainWidget.setLayout(self.mainWidgetLayout)

        # adding the the central widget
        centralWidgetLayout.addWidget(topBar, 0)
        centralWidgetLayout.addWidget(self.mainWidget, 1)
        self.centralWidget.setLayout(centralWidgetLayout)

        #self.setCentralWidget(gridEstimatorImageViewer)
        layoutForThisWidget = QHBoxLayout()
        layoutForThisWidget.setContentsMargins(0,0,0,0)
        layoutForThisWidget.addWidget(self.centralWidget)
        self.setLayout(layoutForThisWidget)
        # self.setCentralWidget(self.centralWidget)

        # self.show()

    def onToggle(self):
        radioButton = self.sender()
        if radioButton.mode == "INDIVIDUAL":
            self.stack.setCurrentIndex(1)
            # self.gridEstimatorImageViewer = IndividualWellImageViewer(QPixmap('wellplate.png'))
            print('Changing it to Individuals')
            # # self.mainWidgetLayout.replaceWidget(
            # #     self.gridEstimatorImageViewer, IndividualWellImageViewer(QPixmap('wellplate.png')))
            # # self.mainWidgetLayout.removeWidget(self.gridEstimatorImageViewer)
            # self.gridEstimatorImageViewer.close()
            # self.gridEstimatorImageViewer.deleteLater()
            # # del self.gridEstimatorImageViewer
            #
            # # self.mainWidget.setLayout(QHBoxLayout())
            # # self.mainWidget.close()
            # self.mainWidget = QWidget()
            # self.mainWidgetLayout = QHBoxLayout()
            # self.mainWidgetLayout.addWidget(IndividualWellImageViewer( QPixmap('wellplage.png') ))
            # self.mainWidget.setLayout( self.mainWidgetLayout)
            # self.mainWidget.update()
            # # self.mainWidgetLayout = QHBoxLayout()
            # self.mainWidgetLayout.update()
            # # self.setCentralWidget(QWidget())
            # self.update()
            # self.setCentralWidget(self.centralWidget)
        else:
            self.stack.setCurrentIndex(0)
            # self.gridEstimatorImageViewer = GridEstimatorImageViewer(QPixmap('wellplate.png'))
            # self.update()

    def changedImage(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        # dlg.setFilter("Text files (*.txt)")
        dlg.exec_()
        filenames = dlg.selectedFiles()
        if len(filenames):
            filename = filenames[0]
            splitText = filenames[0].split('/')
            self.individualImageViewer.setPixmap(QPixmap(filename))
            self.gridEstimatorImageViewer.setPixmap(QPixmap(filename))

    def saveGrid(self):
        if self.stack.currentWidget().grid:

            dlg = QFileDialog()
            filename = dlg.getSaveFileName()
            if filename[0]:
                np.save( filename[0], np.array(self.stack.currentWidget().grid))
            return

    def pressedBack(self, ev):
        self.parent().parent().backPressed()

# end of the define wells program

# Calculate Intrinsic Parameters
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

class IntrinsicParametersWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(IntrinsicParametersWindow, self).__init__(*args, **kwargs)
        self.setMinimumSize(854, 510)
        self.amountOfFishSaved = 0

        # self.vidArray = np.load('vid.npy')
        # self.frameIdx = 0
        # self.amountOfFrames = self.vidArray.shape[-1]

        self.vidArray = None
        self.frameIdx = 0
        self.amountOfFrames = 0

        self.initUI()

    def initUI(self):
        self.setStyleSheet('background: ' + blue + ';')
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
        self.frameSelectionLabel = QLineEdit('0')
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
        self.onFrameLabel = QLabel('On Frame 0 of ' + str(self.amountOfFrames))
        bottomBarLayout.addWidget(self.onFrameLabel, alignment=Qt.AlignHCenter)
        bottomBar.setLayout(bottomBarLayout)

        # self.label = ClickableImageViewer(MyQPixmap('wellplate.png'))
        self.label = ClickableImageViewer(None)
        self.label.setText('No Video Selected')
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # labelContainer = QWidget()
        # labelContainer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # labelContainerLayout = QHBoxLayout()
        # labelContainerLayout.addWidget(self.label, 1, alignment=Qt.AlignCenter)
        # labelContainer.setLayout(labelContainerLayout)

        leftWidget = QWidget()
        # leftWidget.setStyleSheet('border: 1 px solid')
        leftWidgetLayout = QVBoxLayout()
        # leftWidget.setStyleSheet('border: 1px solid')

        # leftWidgetLayout = QGridLayout()
        leftWidgetLayout.addWidget(self.label,1)
        leftWidgetLayout.setStretch(0,1)
        # leftWidgetLayout.addWidget(self.label, 0,0)
        # leftWidgetLayout.addWidget(labelContainer,1)
        leftWidgetLayout.addWidget(bottomBar, 0, alignment=Qt.AlignBottom)

        # leftWidgetLayout.addWidget(bottomBar, 1, 0, 0, 1)
        # leftWidgetLayout.setRowStretch(1,1)

        leftWidgetLayout.setContentsMargins(0,0,0,0)
        leftWidgetLayout.setSpacing(20)
        leftWidget.setLayout(leftWidgetLayout)
        # leftWidget.setStyleSheet('border: 1px solid black')

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
        # mainWidget.setStyleSheet('border: 1px solid')
        mainWidgetLayout = QHBoxLayout()
        # mainWidgetLayout.addWidget(self.label, 200)
        mainWidgetLayout.addWidget(leftWidget, 130)
        mainWidgetLayout.addWidget(sideBar, 40)
        mainWidget.setLayout(mainWidgetLayout)

        # The main widget was created now lets just create the title
        topWidget = QWidget()
        topWidgetLayout = QGridLayout()

        backButton = ImageViewer(QPixmap('Back.png'))
        backButton.mousePressEvent = self.backPressed

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

        # self.setCentralWidget(centralWidget)
        # self.show()

        layoutForThisWidget = QHBoxLayout()
        layoutForThisWidget.setContentsMargins(0,0,0,0)
        layoutForThisWidget.addWidget(centralWidget)
        self.setLayout(layoutForThisWidget)


    def getNewVideo(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.exec_()
        filenames = dlg.selectedFiles()
        if filenames:
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
        if self.amountOfFrames == 0 or self.amountOfFrames == 1: return
        self.frameIdx = (self.frameIdx + 1) % self.amountOfFrames
        self.changeFrame()

    def previousFrame(self):
        if self.amountOfFrames == 0 or self.amountOfFrames == 1: return
        self.frameIdx = (self.frameIdx - 1) % self.amountOfFrames
        self.changeFrame()

    def changeFrame(self):
        if self.amountOfFrames == 0 or self.amountOfFrames == 1: return
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

    def backPressed(self, ev):
        self.parent().parent().backPressed()

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
            # print('drawing points')
            # pen = QPen(Qt.red)
            # pen.setWidth(10)
            qp.setPen( QPen(QColor(0,0,0,0)) )
            qp.setBrush(Qt.cyan)
            for pointIdx, point in enumerate(self.points):
                if pointIdx == 1: qp.setBrush(Qt.red)
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
        self.amountOfCutouts = len(cutouts)

        self.initUI()
        # The following array will be used to quickly get a count of how many fish have been annotated
        self.annotationArray = np.zeros((len(cutouts)))

        for fishIdx, cutout in enumerate(cutouts):
            label = AnnotationsLabel(cutout, 'Fish ' + str(fishIdx + 1) )
            label.mousePressEvent = \
                functools.partial( self.annotationLabelPressed, label)
            #labels.append(labels)
            self.scrollAreaWidgetLayout.addWidget(label)
            # self.scrollAreaWidgetLayout.setStretch(fishIdx, 0)
        self.previousAnnotationLabel = self.scrollAreaWidgetLayout.itemAt(0).widget()
        self.previousAnnotationLabel.setStyleSheet('background: blue')
        # spacer = QWidget()
        # self.scrollAreaWidgetLayout.addWidget(spacer, 1)
        #self.scrollAreaWidgetLayout.setSpacing(0)
        # self.scrollAreaWidgetLayout.setStretchFactor(0)
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
        self.cutoutViewer = ClickableCutoutViewer(QPixmap('temp.png'))
        AnnotationsLabel.imageViewer = self.cutoutViewer
        leftWidgetLayout.addWidget(self.cutoutViewer,1)

        #   Creating the bottom bar
        bottomBar = QWidget()
        bottomBarLayout = QVBoxLayout()
        bottomBarLayout.setContentsMargins(0,0,0,0)
        #   Creating the buttons to iterate through the fish
        buttonsBar = QWidget()
        buttonsBarLayout = QHBoxLayout()
        buttonsBarLayout.setContentsMargins(0,0,0,0)
        backButton = QPushButton('Back')
        backButton.clicked.connect(self.backButtonPressed)
        nextButton = QPushButton('Next')
        nextButton.clicked.connect(self.nextButtonPressed)
        buttonsBarLayout.addWidget(backButton)
        buttonsBarLayout.addWidget(nextButton)
        buttonsBar.setLayout(buttonsBarLayout)
        bottomBarLayout.addWidget(buttonsBar)
        bottomBar.setLayout(bottomBarLayout)

        # TODO: replace this with the amount of labels
        self.onCutoutLabel = QLabel('On Cutout 1 of ' + str(self.amountOfCutouts))
        bottomBarLayout.addWidget(self.onCutoutLabel, alignment=Qt.AlignHCenter)

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
        scrollAreaWidget.setStyleSheet('border: 1px solid')
        self.scrollAreaWidgetLayout = QVBoxLayout()
        self.scrollAreaWidgetLayout.setContentsMargins(0,0,0,0)
        self.scrollAreaWidgetLayout.setSpacing(0)

        scrollAreaWidget.setLayout(self.scrollAreaWidgetLayout)
        scrollArea.setWidget(scrollAreaWidget)
        # scrollArea.setWidget()

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
        saveAnnotationButton.clicked.connect(self.markAnnotationsPressed)
        self.amountSavedLabel = QLabel('Annotations Saved: 0 of ' + str(self.amountOfCutouts))
        rightWidgetBottomTopSectionLayout.addWidget(self.amountSavedLabel, 1, alignment=Qt.AlignHCenter)
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

    def annotationLabelPressed(self, label, event):

        idxOfLabel = int(label.text().split(' ')[-1]) - 1
        self.onCutoutLabel.setText('On Cutout ' + str(idxOfLabel + 1) + ' of ' + str(self.amountOfCutouts))

        if self.previousAnnotationLabel:
            styleSheet = ''
            if self.previousAnnotationLabel.markedPoints:
                styleSheet = 'background: green'
            self.previousAnnotationLabel.setStyleSheet(styleSheet)

        label.setStyleSheet('background: blue')
        cv.imwrite('temp.png', label.cutout)


        self.cutoutViewer.setPixmap(QPixmap('temp.png'))
        if label.markedPoints:
            print('the label had marked points')
            self.cutoutViewer.amountOfPoints = 2
            self.cutoutViewer.points = label.markedPoints
        else:
            self.cutoutViewer.amountOfPoints = 0
            self.cutoutViewer.points = []

        # self.cutoutViewer.setPixmap(QPixmap('temp.png'))

        self.previousAnnotationLabel = label

    def markAnnotationsPressed(self):
        if not self.previousAnnotationLabel: return
        if self.cutoutViewer.amountOfPoints < 2: return
        idxOfLabel = int(self.previousAnnotationLabel.text().split(' ')[-1]) - 1
        self.annotationArray[idxOfLabel] = 1
        amountOfAnnotations = np.count_nonzero(self.annotationArray)
        self.amountSavedLabel.setText('Annotations Saved: ' + str(amountOfAnnotations) + ' of ' + str(self.amountOfCutouts))
        self.previousAnnotationLabel.markedPoints = self.cutoutViewer.points


    def backButtonPressed(self):
        idxOfLabel = int(self.previousAnnotationLabel.text().split(' ')[-1]) - 1
        label = self.scrollAreaWidgetLayout.itemAt((idxOfLabel - 1) % self.amountOfCutouts).widget()
        self.annotationLabelPressed(label, None)

    def nextButtonPressed(self):
        idxOfLabel = int(self.previousAnnotationLabel.text().split(' ')[-1]) - 1
        label = self.scrollAreaWidgetLayout.itemAt( (idxOfLabel + 1) % self.amountOfCutouts).widget()
        self.annotationLabelPressed(label, None)

class AnnotationsLabel(QLabel):

    imageViewer = None

    def __init__(self, cutout, *args, **kwargs):
        super(AnnotationsLabel, self).__init__(*args, **kwargs)
        self.markedPoints = None
        self.cutout = cutout

#


# end of Calculate Intrinsic Parameters


class WellPlateGUI(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(WellPlateGUI, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):

        self.myCentralWidget = QStackedWidget()

        # Creating the pages
        menuPage = MenuPage(self.myCentralWidget)
        predictionWindow = PredictionWindow()
        defineWellsPage = DefineWindowClass()
        intrinsicParametersWindow = IntrinsicParametersWindow()

        # Adding the windows to the pages to the stack
        self.myCentralWidget.addWidget( menuPage )
        self.myCentralWidget.addWidget( predictionWindow )
        self.myCentralWidget.addWidget( defineWellsPage )
        self.myCentralWidget.addWidget( intrinsicParametersWindow )

        self.setCentralWidget(self.myCentralWidget)

        # self.testingWidget = MenuPage()
        # self.testingWidget = PredictionWindow()
        #
        # self.setCentralWidget(self.testingWidget)

        self.setGeometry(300, 300, 600, 600)
        self.show()

    def backPressed(self):
        self.myCentralWidget.setCurrentIndex(0)

    def predictionsPressed(self):
        # print('You Pressed the parents')
        self.myCentralWidget.setCurrentIndex(1)

    def defineWellsPressed(self):
        self.myCentralWidget.setCurrentIndex(2)

    def intrinsicParametersPressed(self):
        self.myCentralWidget.setCurrentIndex(3)

app = QApplication(sys.argv)
ex = WellPlateGUI()
# ex = PredictionWindow()
sys.exit(app.exec_())

















