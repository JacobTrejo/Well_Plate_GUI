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


class DefineWindowClass(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(DefineWindowClass, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        self.setGeometry(300,300,600,400)
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
        self.setCentralWidget(self.centralWidget)
        self.show()

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






class DefineWindowClass2(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(DefineWindowClass2, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 450, 450)

        imageViewer = IndividualWellImageViewer(QPixmap('wellplate.png'))

        self.setCentralWidget(imageViewer)

        self.show()

app = QApplication(sys.argv)
ex = DefineWindowClass()
sys.exit(app.exec())












