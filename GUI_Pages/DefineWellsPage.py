import cv2

from GUI_Pages.Auxilary import *
from GUI_Pages.videoFileExtensions import isVideoFile

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

def makeGridInBounds(grid, imageSizeX, imageSizeY, x_offset, y_offset):
    # TODO: Parse is for centers out of bounds
    grid = np.array(grid)

    grid[:, 0] -= x_offset
    grid[:, 1] -= y_offset

    r = grid[:,2]

    maxXDiff = imageSizeX - (grid[:,0] + r)
    maxYDiff = imageSizeY - (grid[:,1] + r)
    minXDiff = (grid[:,0] - r)
    minYDiff = (grid[:,1] - r)

    minDiff = np.min((maxXDiff, maxYDiff, minXDiff, minYDiff), axis=0)
    minDiffIndices = minDiff < 0


    grid[minDiffIndices,2] = grid[minDiffIndices,2] + minDiff[minDiffIndices]

    grid[:, 0] += x_offset
    grid[:, 1] += y_offset
    # TODO: remove radius of zero

    return grid

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

def estimateGridFrom4Corners(circle1, circle2, circle3, circle4, amountOfRows = None, amountOfColumns = None):
    (x1, y1), r1 = circle1
    (x2, y2), r2 = circle2
    (x3, y3), r3 = circle3
    (x4, y4), r4 = circle4

    sx, bx = min(x1, x4), max(x1, x4)
    sy, by = min(y1, y4), max(y1, y4)

    r = np.mean((r1, r2))

    dx = bx - sx
    dy = by - sy

    if amountOfRows is None: amountOfRows = round(dy / (2 * r)) + 1
    if amountOfColumns is None: amountOfColumns = round(dx / (2 * r)) + 1

    x_direction_rows_first = (x3 - x1) / (amountOfRows - 1)
    y_direction_rows_first = (y3 - y1) / (amountOfRows - 1)

    x_direction_column_first = (x2 - x1) / (amountOfColumns- 1)
    y_direction_column_first = (y2 - y1) / (amountOfColumns- 1)

    x_direction_rows_second = (x4 - x2) / (amountOfRows - 1)
    y_direction_rows_second = (y4 - y2) / (amountOfRows - 1)

    x_direction_column_second = (x4 - x3) / (amountOfColumns- 1)
    y_direction_column_second = (y4 - y3) / (amountOfColumns- 1)


    # stepX = dx / (amountOfColumns - 1)
    # stepY = dy / (amountOfRows - 1)

    grid = []
    for rowIdx in range(amountOfRows):
        for colIdx in range(amountOfColumns):
            x_steps_column = ((amountOfRows - 1 - rowIdx) * x_direction_column_first + (rowIdx) * x_direction_column_second) / (
                        amountOfRows - 1)
            y_steps_column = ((amountOfRows - 1 - rowIdx) * y_direction_column_first + (rowIdx) * y_direction_column_second) / (
                        amountOfRows - 1)

            x = x1 + colIdx * x_steps_column + rowIdx * x_direction_rows_first
            y = y1 + colIdx * y_steps_column + rowIdx * y_direction_rows_first

            grid.append([x, y, r])

    return (grid)

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

        # The labels
        self.xLabel = None
        self.yLabel = None
        self.rLabel = None

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

            if self.parent().parent().parent().parent().showNumbers:
                for circleIdx, circle in enumerate(grid):
                    x, y, r = circle
                    # print(x, y, r)
                    # x, y, r = int(round(x)), int(round(y)), int(round(r))
                    center = QtCore.QPoint(x, y)
                    qp.drawEllipse(center, r, r)
                    qp.drawText(center, str(circleIdx + 1))
            else:
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
            # pen = QPen(Qt.red)
            # pen.setWidth(10)
            qp.setPen( QPen(QColor(0,0,0,0)) )
            qp.setBrush(Qt.red)


            if len(self.points) <= 11:

                for point in self.points:
                    # qp.drawPoint(int(point[0]), int( point[1]) )
                    # qp.drawEllipse(int((point[0] * imWidth) + x_offset ), int( (point[1] * imHeight ) + y_offset ), 5,5)
                    center = QtCore.QPoint( int( round((point[0] * imWidth) + x_offset ) ), int( round((point[1] * imHeight ) + y_offset) ) )
                    qp.drawEllipse(center, 2,2)

                if (len(self.points) / 3) > 0:
                    qp.setPen(Qt.magenta)
                    qp.setBrush(QBrush(QColor(0, 0, 0, 0)))

                    realIdx = 0
                    translatedPoints = np.array(self.points.copy())

                    translatedPoints[:, 0] *= imWidth
                    translatedPoints[:, 0] += x_offset
                    translatedPoints[:, 1] *= imHeight
                    translatedPoints[:, 1] += y_offset
                    amount = int(len(self.points) / 3)

                    for tempIdx in range(amount):
                        realIdx = 3 * tempIdx
                        center, r = returnCircleParameters(translatedPoints[realIdx], translatedPoints[realIdx + 1],
                                                           translatedPoints[realIdx + 2])
                        x, y = center
                        x, y, r = int(round(x)), int(round(y)), int(round(r))
                        center = QtCore.QPoint(x, y)
                        qp.drawEllipse(center, r, r)


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
                # qp.drawEllipse(center, int(round(r)), int(round(r)))

                if len(self.points) == 12:
                    circles = []

                    step = 3
                    for circleIdx in range(4):
                        realIdx = circleIdx * step
                        center, r = returnCircleParameters(translatedPoints[realIdx], translatedPoints[realIdx + 1], translatedPoints[realIdx + 2])
                        circle = (center, r)
                        circles.append(circle)

                    center0, r0 = circles[0]
                    center1, r1 = circles[1]
                    center2, r2 = circles[2]
                    center3, r3 = circles[3]

                    # # Actually we should draw the grid
                    #
                    # # We will draw the grid
                    #
                    # # We will use the remaining three points to draw the second circle
                    # center, r = returnCircleParameters(translatedPoints[3], translatedPoints[4], translatedPoints[5])
                    #
                    # grid = estimateGridFrom2Corners((center0, r0), (center, r))

                    # NOTE: the following two lines depend on the parent
                    rowsText = self.parent().parent().parent().parent().rowsInput.text()
                    columnsText = self.parent().parent().parent().parent().columnsInput.text()

                    if rowsText.isnumeric() and columnsText.isnumeric():
                        rowsText = int(rowsText)
                        columnsText = int(columnsText)
                        grid = estimateGridFrom4Corners((center0, r0), (center1, r1),
                                                        (center2, r2), (center3, r3),
                                                        rowsText, columnsText)

                    else:
                        grid = estimateGridFrom4Corners((center0, r0), (center1, r1),
                                                        (center2, r2), (center3, r3))

                    grid = makeGridInBounds(grid, imWidth, imHeight, x_offset, y_offset)

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
            self.amountOfPoints = (self.amountOfPoints + 1) % 13

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

            # Setting the labels of the selected object:
            x0, y0, r0 = self.grid[self.selectedObject[1]]
            self.xLabel.setText('x: ' + str(int(x0 * self.arrayShape[1])))
            self.yLabel.setText('y: ' + str(int(y0 * self.arrayShape[1])))
            self.rLabel.setText('r: ' + str(int(r0 * self.arrayShape[1])))

            if self.selectedObject[0]:
                x0, y0, r0 = self.grid[self.selectedObject[1]]
                r = ((x - x0)**2 + (y - y0)**2)**.5
                if (y0 - r) * (imWidth / imHeight) <= 0 or (y0 + r) * (imWidth / imHeight) >= 1 or (x0 - r) <= 0 or (x0 + r) >= 1: return
                self.grid[self.selectedObject[1]] = x0, y0, r
            else:
                r = self.grid[self.selectedObject[1]][-1]
                if (y - r) * (imWidth / imHeight) <= 0 or (y + r) * (imWidth / imHeight) >= 1 or (x - r) <= 0 or (x + r) >= 1: return
                self.grid[self.selectedObject[1]] = x, y, self.grid[self.selectedObject[1]][-1]
            self.update()
            return

        imHeight, imWidth = self.scaled.height(), self.scaled.width()

        x_offset = (self.width() - imWidth) / 2
        y_offset = (self.height() - imHeight) / 2

        selectedObject = getSelectedObject((ev.x(), ev.y()), self.grid, (x_offset, y_offset), (imWidth, imHeight) )

        if selectedObject:
            self.selectedObject = selectedObject
            x0, y0, r0 = self.grid[self.selectedObject[1]]
            self.xLabel.setText('x: ' + str(int(x0 * self.arrayShape[1])))
            self.yLabel.setText('y: ' + str(int(y0 * self.arrayShape[1])))
            self.rLabel.setText('r: ' + str(int(r0 * self.arrayShape[1])))
            self.update()
        else:
            if self.selectedObject:
                self.selectedObject = None
                self.xLabel.setText('x: ')
                self.yLabel.setText('y: ')
                self.rLabel.setText('r: ')
                self.update()
            # self.selectedObject = None

        #print('X: ', ev.x())
        #print('Y: ', ev.y())
    def setLabels(self, labels):
        self.xLabel, self.yLabel, self.rLabel = labels

    def setArrayShape(self, shape):
        self.arrayShape = shape

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

        # The labels
        self.xLabel = None
        self.yLabel = None
        self.rLabel = None
        self.arrayShape = None

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

            if self.parent().parent().parent().parent().showNumbers:
                for circleIdx, circle in enumerate(grid):
                    x, y, r = circle
                    # print(x, y, r)
                    # x, y, r = int(round(x)), int(round(y)), int(round(r))
                    center = QtCore.QPoint(x, y)
                    qp.drawEllipse(center, r, r)
                    qp.drawText(center,str(circleIdx + 1))
            else:
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
                (x0, y0), r0 = circle
                x0, y0 = x0 - x_offset, y0 - y_offset
                if x0 <= 0 or y0 <= 0: return
                minDiff = np.min([imWidth - (x0 + r0), imHeight - (y0 + r0), x0 - r0, y0 - r0])
                if minDiff < 0:
                    r0 += minDiff
                    circle = (x0 + x_offset, y0 + y_offset), r0
                # circle = [center[0], center[1], radius]
                normalizedCircle = normalizeCircle(circle, (x_offset, y_offset), (imWidth, imHeight))
                self.grid.append(normalizedCircle)

                # self.grid.append(circle)
                # self.grid = normalizeGrid(self.grid, (x_offset, y_offset), (imWidth, imHeight))

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

            x0, y0, r0 = self.grid[self.selectedObject[1]]
            self.xLabel.setText('x: ' + str(int(x0 * self.arrayShape[1])))
            self.yLabel.setText('y: ' + str(int(y0 * self.arrayShape[1])))
            self.rLabel.setText('r: ' + str(int(r0 * self.arrayShape[1])))

            if self.selectedObject[0]:
                x0, y0, r0 = self.grid[self.selectedObject[1]]
                r = ((x - x0)**2 + (y - y0)**2)**.5
                if (y0 - r) * (imWidth / imHeight) <= 0 or (y0 + r) * (imWidth / imHeight) >= 1 or (x0 - r) <= 0 or (x0 + r) >= 1: return
                self.grid[self.selectedObject[1]] = x0, y0, r
            else:
                r = self.grid[self.selectedObject[1]][-1]
                if (y - r) * (imWidth / imHeight) <= 0 or (y + r) * (imWidth / imHeight) >= 1 or (x - r) <= 0 or (x + r) >= 1: return
                self.grid[self.selectedObject[1]] = x, y, self.grid[self.selectedObject[1]][-1]
            self.update()
            return

        imHeight, imWidth = self.scaled.height(), self.scaled.width()

        x_offset = (self.width() - imWidth) / 2
        y_offset = (self.height() - imHeight) / 2

        selectedObject = getSelectedObject((ev.x(), ev.y()), self.grid, (x_offset, y_offset), (imWidth, imHeight) )

        if selectedObject:
            self.selectedObject = selectedObject
            x0, y0, r0 = self.grid[self.selectedObject[1]]
            self.xLabel.setText('x: ' + str(int(x0 * self.arrayShape[1])))
            self.yLabel.setText('y: ' + str(int(y0 * self.arrayShape[1])))
            self.rLabel.setText('r: ' + str(int(r0 * self.arrayShape[1])))
            self.update()
        else:
            if self.selectedObject:
                self.selectedObject = None
                self.xLabel.setText('x: ')
                self.yLabel.setText('y: ')
                self.rLabel.setText('r: ')
                self.update()
            # self.selectedObject = None

    def removeCircle(self, index):
        self.grid.pop(index)
        self.selectedObject = None
        self.update()
        #print('X: ', ev.x())
        #print('Y: ', ev.y())

    def setLabels(self, labels):
        self.xLabel, self.yLabel, self.rLabel = labels

    def setArrayShape(self, shape):
        self.arrayShape = shape

class TitleLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super(TitleLabel, self).__init__( *args, **kwargs)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    def resizeEvent(self, a0):
        font = self.font()
        font.setPixelSize( int( self.height() * .7))
        self.setFont(font)

class DefineWellsPage(QWidget):

    def __init__(self, *args, **kwargs):
        super(DefineWellsPage, self).__init__(*args, **kwargs)
        self.showNumbers = False
        self.initUI()

    def initUI(self):
        self.setStyleSheet('background: ' + blue + ';')
        # self.setGeometry(300,300,600,400)

        # gridEstimatorImageViewer = GridEstimatorImageViewer(QPixmap('wellplate.png'))

        self.centralWidget = QWidget(self)
        # centralWidget.setStyleSheet('border: 1px solid')
        centralWidgetLayout = QVBoxLayout()
        centralWidgetLayout.setSpacing(20)
        centralWidgetLayout.setContentsMargins(10,20,10,10)
        # Creating the top bar
        topBar = QWidget()
        # topBar.setStyleSheet('border: 1px solid')
        topBarLayout = QGridLayout()
        topBarLayout.setContentsMargins(0,0,0,0)
        backButton = QLabel("back")
        backButton.mousePressEvent = self.pressedBack

        title = QLabel("Define Wells")
        title = TitleLabel('Define Wells')
        title.setMinimumSize(100, 50)
        title.setMaximumHeight(70)
        title.setStyleSheet('color: ' + firstColorName)
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
        self.gridEstimatorImageViewer.setStyleSheet('border: 1px solid')
        self.individualImageViewer.setStyleSheet('border: 1px solid')
        self.gridEstimatorImageViewer.setAlignment(Qt.AlignCenter)
        self.individualImageViewer.setAlignment(Qt.AlignCenter)
        self.gridEstimatorImageViewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.stack.addWidget(self.gridEstimatorImageViewer)
        self.stack.addWidget(self.individualImageViewer)
        # self.stack.setStyleSheet('border: 1px solid')
        #   Creating the right side, the sidebar
        sideBar = QWidget()

        # sideBar.setStyleSheet('border: 1px solid')
        sideBarLayout = QVBoxLayout()
        sideBarLayout.setContentsMargins(0,0,0,0)

        scrollArea = QScrollArea()

        topFrame = QFrame()
        topFrame.setStyleSheet('border: 1px solid')
        topFrameLayout = QVBoxLayout()
        topFrameLayout.setContentsMargins(0,0,0,0)
        topFrameLayout.setSpacing(0)
        topFrameTitle = QLabel('Info')
        topFrameTitle.setStyleSheet('border: 0px;'+
                                    'border-bottom: 1px solid')
        topFrameTitle.setAlignment(Qt.AlignHCenter)
        topFrameLayout.addWidget(topFrameTitle, 0)
        topInfo = QWidget()
        topInfo.setStyleSheet('border: 0px')
        topInfoLayout = QHBoxLayout()
        topInfoLayout.setContentsMargins(5,0,0,0)
        topInfoLayout.setSpacing(0)
        self.xInfo = QLabel('x: ')
        self.yInfo = QLabel('y: ')
        topInfoLayout.addWidget(self.xInfo)
        topInfoLayout.addWidget(self.yInfo)
        topInfo.setLayout(topInfoLayout)
        self.rInfo = QLabel('r: ')
        self.rInfo.setStyleSheet('border: 0px;' +
                                 'border-top: 1px solid')
        topFrameLayout.addWidget(topInfo, 1)
        topFrameLayout.addWidget(self.rInfo, 1)
        topFrame.setLayout(topFrameLayout)

        self.gridEstimatorImageViewer.setLabels([self.xInfo, self.yInfo, self.rInfo])
        self.individualImageViewer.setLabels([self.xInfo, self.yInfo, self.rInfo])
        # topFrame = QTextEdit()
        # topFrame.setReadOnly(True)
        # topFrame.setMinimumSize(0,0)
        # topFrame.setStyleSheet('border: 1px solid')


        middleFrame = QWidget()
        middleFrame.setObjectName('middleFrame')
        middleFrame.setStyleSheet('QWidget#middleFrame {border: 1px solid;}')
        middleFrameLayout = QVBoxLayout()
        middleFrameLayout.setContentsMargins(1,1,1,1)
        middleFrameLayout.setSpacing(0)
        middleFrameTitle = QLabel('Settings')
        middleFrameTitle.setStyleSheet('border-bottom: 1px solid;')
        middleFrameTitle.setAlignment(Qt.AlignHCenter)

        radioButtonsRow = QWidget()
        radioButtonsRow.setObjectName('radioButtonsRow')
        radioButtonsRow.setStyleSheet('QWidget#radioButtonsRow {border-bottom: 1px solid}')
        radioButtonsRowLayout = QVBoxLayout()
        radioButtonsRowLayout.setContentsMargins(0,0,0,1)
        radioButtonsRowLayout.setSpacing(0)

        radioButtonsRowTitle = QLabel('Mode')
        # radioButtonsRowTitle.setStyleSheet('border-left: 1px solid;' +
        #                                    'border-right: 1px solid;')

        radioButtons = QWidget()
        # radioButtons.setStyleSheet('border: 0px')
        radioButtonsLayout = QHBoxLayout()
        radioButtonsLayout.setContentsMargins(10,0,10,0)

        estimateButton = QRadioButton('Estimate')
        estimateButton.setChecked(True)
        estimateButton.mode = 'ESTIMATE'
        estimateButton.toggled.connect(self.onToggle)
        radioButtonsLayout.addWidget(estimateButton)

        individualButton = QRadioButton('Individual')
        individualButton.mode = 'INDIVIDUAL'
        individualButton.toggled.connect(self.onToggle)
        # individualButton.setStyleSheet('border: 0px')
        radioButtonsLayout.addWidget(individualButton)
        radioButtons.setLayout(radioButtonsLayout)

        radioButtonsRowLayout.addWidget(radioButtonsRowTitle, 1)
        radioButtonsRowLayout.addWidget(radioButtons, 3)
        radioButtonsRow.setLayout(radioButtonsRowLayout)

        # Hint
        hintsRow = QWidget()
        hintsRow.setObjectName('hintsRow')
        hintsRow.setStyleSheet('QWidget#hintsRow {border-bottom: 1px solid;}')
        hintsRowLayout = QVBoxLayout()
        hintsRowLayout.setContentsMargins(0,0,0,1)
        hintsRowLayout.setSpacing(0)
        # hintsRowLayout.setStyleSheet('border: 0px')

        hintLabel = QLabel('Hint')
        # hintLabel.setStyleSheet('border: 1px solid;')

        hintsInputs = QWidget()
        hintsInputsLayout = QHBoxLayout()
        hintsInputsLayout.setContentsMargins(0,0,0,0)

        rowsLabel = QLabel('Rows: ')
        # rowsLabel.setStyleSheet('border: 0px;' +
        #                         'border-bottom: 1px solid;'+
        #                         'border-left: 1px solid;')
        self.rowsInput = QLineEdit(' ')

        columnsLabel = QLabel('Columns: ')
        # columnsLabel.setStyleSheet('border: 0px;' +
        #                'border-bottom: 1px solid;')
        self.columnsInput = QLineEdit(' ')
        # columnsInput.setStyleSheet('border: 0px;' +
        #                         'border-bottom: 1px solid;'+
        #                         'border-right: 1px solid;')
        hintsInputsLayout.addWidget(rowsLabel)
        hintsInputsLayout.addWidget(self.rowsInput)
        hintsInputsLayout.addWidget(columnsLabel)
        hintsInputsLayout.addWidget(self.columnsInput)
        hintsInputs.setLayout(hintsInputsLayout)

        hintsRowLayout.addWidget(hintLabel, 1)
        hintsRowLayout.addWidget(hintsInputs, 3)
        hintsRow.setLayout(hintsRowLayout)

        # The button for toggling
        toggleButtonWrapper = QWidget()
        # toggleButtonWrapper.setStyleSheet('border: 0px;' +
        #                                   'border-bottom: 1px solid;')
        toggleButtonWrapperLayout = QVBoxLayout()
        # toggleButtonWrapperLayout.setStyleSheet('border: 0px')

        self.toggleButton = QPushButton(' Show Numbers ')
        self.toggleButton.setStyleSheet(smallerButtonStyleSheet)
        toggleButtonWrapperLayout.addWidget(self.toggleButton)
        self.toggleButton.clicked.connect(self.showNumbersButton)
        toggleButtonWrapper.setLayout(toggleButtonWrapperLayout)

        middleFrameLayout.addWidget(middleFrameTitle, 0)
        middleFrameLayout.addWidget(radioButtonsRow, 2)
        middleFrameLayout.addWidget(hintsRow, 2)
        middleFrameLayout.addWidget(toggleButtonWrapper, 1, alignment = Qt.AlignHCenter)

        # middleFrame.setStyleSheet('border: 1px solid')
        middleFrame.setLayout(middleFrameLayout)

        bottomFrame = QWidget()
        bottomFrame.setStyleSheet('border: 1px solid')
        bottomFrameLayout = QVBoxLayout()
        saveGridButton = QPushButton('Save Grid')
        saveGridButton.clicked.connect(self.saveGrid)
        saveGridButton.setStyleSheet(smallerButtonStyleSheet)
        changeVidButton = QPushButton('Change Image')
        changeVidButton.clicked.connect( self.changedImage )
        changeVidButton.setStyleSheet(smallerButtonStyleSheet)
        bottomFrameLayout.addWidget(changeVidButton)
        bottomFrameLayout.addWidget(saveGridButton)
        bottomFrame.setLayout(bottomFrameLayout)

        # bottomFrame.setStyleSheet('border: 1px solid')

        sideBarLayout.addWidget(topFrame, 1)
        sideBarLayout.addWidget(middleFrame, 1)
        sideBarLayout.addWidget(bottomFrame, 1)

        sideBar.setLayout(sideBarLayout)

        # adding to main widget

        # self.mainWidgetLayout.addWidget(self.gridEstimatorImageViewer, 5)
        # self.mainWidgetLayout.addWidget(self.stack, 5, alignment = Qt.AlignCenter)
        self.mainWidgetLayout.addWidget(self.stack, 5)

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
    def showNumbersButton(self):
        self.showNumbers = not self.showNumbers
        self.gridEstimatorImageViewer.update()
        self.individualImageViewer.update()

        if self.showNumbers:
            self.toggleButton.setStyleSheet(invertedSmallerButtonStyleSheet)
        else:
            self.toggleButton.setStyleSheet(smallerButtonStyleSheet)

    def onToggle(self):
        radioButton = self.sender()
        if radioButton.mode == "INDIVIDUAL":
            self.stack.setCurrentIndex(1)
            self.rowsInput.setEnabled(False)
            self.columnsInput.setEnabled(False)
            # self.gridEstimatorImageViewer = IndividualWellImageViewer(QPixmap('wellplate.png'))
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
            self.rowsInput.setEnabled(True)
            self.columnsInput.setEnabled(True)
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
            try:
                # this line will allow us to check if an image was selected
                if isVideoFile(filename):
                    vidObj = cv.VideoCapture(filename)
                    ret, frame0 = vidObj.read()
                    arrayShape = frame0.shape[:2]
                    cv.imwrite('temp.png', frame0)
                    filename = 'temp.png'
                else:
                    arrayShape = cv.imread(filename).shape[:2]

            except:
                rect = QtCore.QRect(QCursor.pos().x(), QCursor.pos().y(), 140, 50)
                QToolTip.showText(QCursor.pos(), 'Expected an image or video file', None, rect, 3000)
                return
            splitText = filenames[0].split('/')
            self.individualImageViewer.setPixmap(QPixmap(filename))
            self.gridEstimatorImageViewer.setPixmap(QPixmap(filename))

            self.gridEstimatorImageViewer.setArrayShape(arrayShape)
            self.individualImageViewer.setArrayShape(arrayShape)

            self.xInfo.setText('x: ')
            self.yInfo.setText('y: ')
            self.rInfo.setText('r: ')

    def saveGrid(self):
        if self.stack.currentWidget().grid:

            dlg = QFileDialog()
            filename = dlg.getSaveFileName()
            if filename[0]:
                np.save( filename[0], np.array(self.stack.currentWidget().grid))
            return

    def pressedBack(self, ev):
        self.parent().parent().backPressed()


if __name__ == '__main__':
    from Testing import *
    TestingWindow.testingClass = DefineWellsPage
    run()




