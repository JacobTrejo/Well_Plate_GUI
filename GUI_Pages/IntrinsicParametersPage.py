import time

import cv2
from scipy.stats import norm
from GUI_Pages.Auxilary import *
import os
from GUI_Pages.videoFileExtensions import isVideoFile
from GUI_Pages.bgsub import *
from GUI_Pages.IntrinsicParametersFunctions.calculate_intrinsic_parameters import calculate_intrinsic_parameters

class TitleLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super(TitleLabel, self).__init__( *args, **kwargs)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    def resizeEvent(self, a0):
        font = self.font()
        font.setPixelSize( int( self.height() * .7))
        self.setFont(font)

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
        # Information about the label, usefull for bgsub
        self.url = None
        self.frameIdx = None
        self.crops = None # sy, by, sx, bx

        FishLabel.counter += 1
    @ property
    def bgsubInfo(self):
        return (self.url, self.frameIdx, self.crops)

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
                cv.imwrite('GUI_Pages/temp.png', self.getCutout()[0])
                self.cutoutWidget.setPixmap(QPixmap('GUI_Pages/temp.png'))
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

        return self.pixmap.arr[sy: by + 1,sx: bx + 1], [sy, by, sx, bx]

class Video:
    """
        A class to make is easier to handle when videos are actual videos vs folders with pictures
    """
    # Video types
    actualVideo = 'actualVideo'
    folderVideo = 'folderVideo'
    # initially the type is NONE

    def __init__(self):
        self.isVideoLoaded = False
        self.videoType = None
        self.video = None
        self.amountOfFrames = None
        self.url = None
    def setVideo(self, url):
        wasItLoadedSuccesfully = False

        # let's check what it is
        if isVideoFile(url):
            # let's assume it actually is a video
            try:
                self.video = cv.VideoCapture(url)
                self.amountOfFrames = int(self.video.get(cv.CAP_PROP_FRAME_COUNT))
                self.videoType = Video.actualVideo
                self.url = url
                wasItLoadedSuccesfully = True
            except:
                wasItLoadedSuccesfully = False
        elif os.path.isdir(url):
            self.video = os.listdir(url)
            self.video.sort()
            try:
                cv.imread(url + self.video[0])
                self.amountOfFrames = len(self.video)
                self.videoType = Video.folderVideo
                self.url = url
                wasItLoadedSuccesfully = True
            except:
                wasItLoadedSuccesfully = False
        # NOTE: you might want to send a message if it was not loaded successfully
    def getFrame(self, frameIdx):

        if self.videoType == Video.actualVideo:
            self.video.set(cv.CAP_PROP_POS_FRAMES, frameIdx)
            ret, frame = self.video.read()
            return frame

        elif self.videoType == Video.folderVideo:
            filename = self.url + '/' + self.video[frameIdx]
            frame = cv.imread(filename)
            return frame

class IntrinsicParametersPage(QWidget):
    def __init__(self, *args, **kwargs):
        super(IntrinsicParametersPage, self).__init__(*args, **kwargs)
        self.setMinimumSize(854, 510)
        self.amountOfFishSaved = 0

        # self.vidArray = np.load('vid.npy')
        # self.frameIdx = 0
        # self.amountOfFrames = self.vidArray.shape[-1]

        # Video object to handle the different formats of video
        self.video = Video()

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
        self.label.setStyleSheet('border: 1px solid')
        self.label.setAlignment(Qt.AlignCenter)

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
        sideBarLayout.setContentsMargins(0,0,0,0)
        frame1 = QFrame()
        frame1.setStyleSheet('border: 1px solid')
        frame1Layout = QGridLayout()
        cutoutLabel = QLabel('Cutout')
        cutoutLabel.setStyleSheet('border: 0px')
        frame1Layout.addWidget(cutoutLabel, 0,0,alignment=Qt.AlignHCenter)
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
        frame2.setStyleSheet('border: 1px solid')
        frame2Layout = QVBoxLayout()
        saveCutoutButton = QPushButton('Save Cutout')
        saveCutoutButton.clicked.connect( self.pressedSaveFish )
        saveCutoutButton.setStyleSheet(smallerButtonStyleSheet)
        saveCutoutButton.setMinimumWidth(130)
        # saveCutoutButton.setStyleSheet('border: 0px')

        self.cutoutsSavedLabel = QLabel('Amount Saved: 0')
        self.cutoutsSavedLabel.setStyleSheet('border: 0px')
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
        frame4.setStyleSheet('border: 1px solid')
        frame4Layout = QVBoxLayout()
        startAnnotatingButton = QPushButton('Start Annotating')
        startAnnotatingButton.clicked.connect(self.startAnnotating)
        startAnnotatingButton.setStyleSheet(smallerButtonStyleSheet)
        startAnnotatingButton.setMinimumWidth(130)
        changeVideoButton = QPushButton('Change Video')
        changeVideoButton.clicked.connect( self.getNewVideo )
        changeVideoButton.setStyleSheet(smallerButtonStyleSheet)
        changeVideoButton.setMinimumWidth(130)

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

        # backButton = ImageViewer(QPixmap('Back.png'))
        backButton = QLabel('back')
        backButton.mousePressEvent = self.backPressed

        # title = QLabel('Intrinsic Parameters')
        title = TitleLabel('Intrinsic Parameters')
        title.setMinimumSize(100, 50)
        title.setMaximumHeight(70)
        title.setStyleSheet('color: ' + firstColorName)
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
            # Checking what type the file is
            filename = filenames[0]
            if os.path.isfile(filename):
                print('You loaded a file')
            elif os.path.isdir(filename):
                print('You loaded a folder')
            self.video.setVideo(filename)
            # self.vidArray = np.load(filenames[0])
            self.frameIdx = 0
            # self.amountOfFrames = self.vidArray.shape[-1]
            self.amountOfFrames = self.video.amountOfFrames
            self.changeFrame()
            self.cutOutWidget.setPixmap(None)
            self.label.amountOfPoints = 0
            self.label.points = []
    def pressedSaveFish(self):
        if self.label.amountOfPoints == 2:
            self.amountOfFishSaved += 1
            cutout, crops = self.label.getCutout()
            fishLabel = FishLabel('Fish ' + str(self.amountOfFishSaved))
            fishLabel.cutout = cutout
            # adding the information for bgsub
            fishLabel.frameIdx = self.frameIdx
            fishLabel.url = self.video.url
            fishLabel.crops = crops
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
        cv.imwrite('GUI_Pages/temp.png', fishLabel.cutout)
        self.cutOutWidget.setPixmap(QPixmap('GUI_Pages/temp.png'))
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
        frame = self.video.getFrame(self.frameIdx)
        # frame = self.vidArray[..., self.frameIdx]
        cv.imwrite('GUI_Pages/temp.png', frame)
        self.label.amountOfPoints = 0
        self.label.points = []
        self.label.setPixmap(MyQPixmap('GUI_Pages/temp.png'))

    def startAnnotating(self):
        # print('amount of labels: ', self.vBoxForScrollArea.count())
        cutouts = []
        bgsubInfoList = []
        for idx in range(self.vBoxForScrollArea.count()):
            cutout = self.vBoxForScrollArea.itemAt(idx).widget().cutout
            bgsubInfo = self.vBoxForScrollArea.itemAt(idx).widget().bgsubInfo
            cutouts.append(cutout)
            bgsubInfoList.append(bgsubInfo)
        dialogWindow = AnnotationsDialog(cutouts, bgsubInfoList)
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
        # print('x: ', x)
        # print('y: ', y)
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

    def __init__(self, cutouts, bgsubInfoList,*args, **kwargs):
        super(AnnotationsDialog, self).__init__(*args, **kwargs)
        self.cutouts = cutouts
        self.amountOfCutouts = len(cutouts)
        self.amountOfDataPutInCutOutData = 0
        self.initUI()
        # The following array will be used to quickly get a count of how many fish have been annotated
        self.annotationArray = np.zeros((len(cutouts)))

        for fishIdx, cutout in enumerate(cutouts):
            label = AnnotationsLabel(cutout, bgsubInfoList[fishIdx],'Fish ' + str(fishIdx + 1) )
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
        self.setStyleSheet('background: ' + blue)
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
        cv.imwrite('GUI_Pages/temp.png', cutout0)
        self.cutoutViewer = ClickableCutoutViewer(QPixmap('GUI_Pages/temp.png'))
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
        saveAnnotationButton.setStyleSheet(smallerButtonStyleSheet)
        self.amountSavedLabel = QLabel('Annotations Saved: 0 of ' + str(self.amountOfCutouts))
        rightWidgetBottomTopSectionLayout.addWidget(self.amountSavedLabel, 1, alignment=Qt.AlignHCenter)
        rightWidgetBottomTopSectionLayout.addWidget(saveAnnotationButton, 1)
        rightWidgetBottomTopSection.setLayout(rightWidgetBottomTopSectionLayout)

        doneButton = QPushButton('Done')
        doneButton.clicked.connect(self.doneFunction)
        doneButton.setStyleSheet(smallerButtonStyleSheet)
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
        firstPage = QWidget()
        centralWidgetLayout.addWidget(mainWidget)
        firstPage.setLayout(centralWidgetLayout)

        # Creating the second Page
        secondPage = QWidget()
        secondPageLayout = QVBoxLayout()
        secondPageTitle = QLabel('Annotation Options')
        secondPageTitle.setAlignment(Qt.AlignHCenter)
        secondPageLayout.addWidget(secondPageTitle, 0)

        secondPageButtons = QWidget()
        secondPageButtonsLayout = QHBoxLayout()
        calculateButton = QPushButton('Calculate Now')
        calculateButton.clicked.connect(self.calculate)
        calculateButton.setStyleSheet(smallerButtonStyleSheet)
        dataButton = QPushButton('Save Data')
        dataButton.clicked.connect(self.saveData)
        dataButton.setStyleSheet(smallerButtonStyleSheet)
        secondPageButtonsLayout.addWidget(calculateButton)
        secondPageButtonsLayout.addWidget(dataButton)
        secondPageButtons.setLayout(secondPageButtonsLayout)
        secondPageLayout.addWidget(secondPageButtons, 1)

        secondPage.setLayout(secondPageLayout)

        # Creating the third Page
        thirdPage = QWidget()
        thirdPageLayout = QVBoxLayout()
        thirdPageTitle = QLabel('Progress')
        thirdPageTitle.setAlignment(Qt.AlignHCenter)
        thirdPageLayout.addWidget(thirdPageTitle, 0)

        self.progressBar = QProgressBar()
        thirdPageLayout.addWidget(self.progressBar, alignment = Qt.AlignVCenter)
        thirdPage.setLayout(thirdPageLayout)

        # Adding the pages to our annotations window, aka self
        self.stackWidget = QStackedWidget()
        self.stackWidget.addWidget(firstPage)
        self.stackWidget.addWidget(secondPage)
        self.stackWidget.addWidget(thirdPage)

        myLayout = QVBoxLayout()
        myLayout.addWidget(self.stackWidget,1)
        self.setLayout(myLayout)

        # centralWidgetLayout.addWidget(self.stackWidget, 1)
        #
        # # centralWidgetLayout.addWidget(mainWidget, 1)
        #
        # self.setLayout(centralWidgetLayout)
        #
        # # dialogWindow.exec_()

    def annotationLabelPressed(self, label, event):

        idxOfLabel = int(label.text().split(' ')[-1]) - 1
        self.onCutoutLabel.setText('On Cutout ' + str(idxOfLabel + 1) + ' of ' + str(self.amountOfCutouts))

        if self.previousAnnotationLabel:
            styleSheet = ''
            if self.previousAnnotationLabel.markedPoints:
                styleSheet = 'background: green'
            self.previousAnnotationLabel.setStyleSheet(styleSheet)

        label.setStyleSheet('background: blue')
        cv.imwrite('GUI_Pages/temp.png', label.cutout)


        self.cutoutViewer.setPixmap(QPixmap('GUI_Pages/temp.png'))
        if label.markedPoints:
            print('the label had marked points')
            self.cutoutViewer.amountOfPoints = 2
            self.cutoutViewer.points = label.markedPoints
        else:
            self.cutoutViewer.amountOfPoints = 0
            self.cutoutViewer.points = []

        # self.cutoutViewer.setPixmap(QPixmap('temp.png'))

        self.previousAnnotationLabel = label

    def doneFunction(self):
        self.stackWidget.setCurrentIndex(1)
        return
        if np.count_nonzero(self.annotationArray) < self.amountOfCutouts: return

    def saveData(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        # dlg.setFilter("Text files (*.txt)")
        dlg.exec_()
        filenames = dlg.selectedFiles()
        if len(filenames):
            folderName = filenames[0]
            splitText = filenames[0].split('/')
            print('The Folder you want to save to is: ', splitText[-1])
            if not os.path.isdir(folderName):
                print('You need to create a folder')
                return
        else:
            return

        imagesFolder = folderName + '/images/'
        pointsFolder = folderName + '/points/'
        os.makedirs(pointsFolder)
        os.makedirs(imagesFolder)

        amountOfLabels = self.scrollAreaWidgetLayout.count()
        videos = []
        videoNames = []
        for labelIdx in range(amountOfLabels):
            bgsubInfo = self.scrollAreaWidgetLayout.itemAt(labelIdx).widget().bgsubInfo
            points = self.scrollAreaWidgetLayout.itemAt(labelIdx).widget().markedPoints
            videoName = bgsubInfo[0]
            if videoName not in videoNames:
                videoNames.append(videoName)
                videos.append([(*bgsubInfo, points)])
            else:
                idx = videoName.index(videoName)
                videos[idx].append((*bgsubInfo, points))
        data = []
        for videoName, vid in zip(videoNames, videos):
            if os.path.isfile(videoName):
                # it is a video, is validity was check when loading

                newData = self.getDataFromVideo(vid)
                data = data + newData
            elif os.path.isdir(videoName):
                newData = self.getDataFromFolder(vid)
                data = data + newData

        for el in data:
            cutout, points = el
            # cv.imwrite('outputs/cutout_data/images/fish_' + str(self.amountOfDataPutInCutOutData) + '.png', cutout)
            # np.save('outputs/cutout_data/points/fish_' + str(self.amountOfDataPutInCutOutData) + '.npy', np.array(points))

            cv.imwrite(imagesFolder + 'fish_' + str(self.amountOfDataPutInCutOutData) + '.png', cutout )
            np.save(pointsFolder + 'fish_' + str(self.amountOfDataPutInCutOutData) + '.npy', np.array(points))

            self.amountOfDataPutInCutOutData += 1
        self.amountOfDataPutInCutOutData = 0


    def calculate(self):
        self.stackWidget.setCurrentIndex(2)
        # Choosing the location in which to save your yaml file
        dlg = QFileDialog()
        # dlg.setFileMode(QFileDialog.AnyFile)
        # # dlg.setFilter("Text files (*.txt)")
        # dlg.exec_()
        filenames = dlg.getSaveFileName()

        if len(filenames):
            filename = filenames[0]

        else:
            return


        # Getting the data
        amountOfLabels = self.scrollAreaWidgetLayout.count()
        videos = []
        videoNames = []
        for labelIdx in range(amountOfLabels):
            bgsubInfo = self.scrollAreaWidgetLayout.itemAt(labelIdx).widget().bgsubInfo
            points = self.scrollAreaWidgetLayout.itemAt(labelIdx).widget().markedPoints
            videoName = bgsubInfo[0]
            if videoName not in videoNames:
                videoNames.append(videoName)
                videos.append([(*bgsubInfo, points)])
            else:
                idx = videoName.index(videoName)
                videos[idx].append((*bgsubInfo, points))
        data = []
        for videoName, vid in zip(videoNames, videos):
            if os.path.isfile(videoName):
                # it is a video, is validity was check when loading

                newData = self.getDataFromVideo(vid)
                data = data + newData
            elif os.path.isdir(videoName):
                newData = self.getDataFromFolder(vid)
                data = data + newData
        # Getting the calculations on the data
        xVectors = np.array([])
        seglens = []
        amountOfData = len(data)
        self.progressBar.setRange(0, amountOfData)

        for dataIdx, el in enumerate(data):
            (cutout, points) = el
            points = np.array(points)
            x, seglen = calculate_intrinsic_parameters(cutout, points)
            self.progressBar.setValue(dataIdx + 1)
            xVectors = np.vstack((xVectors, x)) if xVectors.size > 0 else x
            seglens.append(seglen)
        seglens = np.array(seglens)

        # Turning the data into the yaml file
        modelValueNames = ['d_eye', 'c_eye', 'c_belly', 'c_head', 'eye_br', 'belly_br', 'head_br', 'eye_w', 'eye_l',
                           'belly_w', 'belly_l', 'head_w', 'head_l', 'ball_size', 'ball_thickness', 'tail_br', 'seglen']
        modelValues = []
        ip = xVectors[:,3:]

        for pIdx in range(ip.shape[1] + 1):
            if pIdx == ip.shape[1]:
                values = seglens
            else:
                values = ip[:, pIdx]

            mu, std = norm.fit(values)
            modelValues.append([mu, std])

        # Actually writing the file
        ipFile = open(filename, 'w')

        for parameterValues, parameterName in zip(modelValues, modelValueNames):
            mu, std = parameterValues

            # Writting the distribution
            distributionLine = parameterName + '_distribution: np.random.normal('
            distributionLine += str(mu) + ' ,' + str(std) + ' )\n'
            ipFile.write(distributionLine)

            # Writting the mean
            meanLine = parameterName + '_u: ' + str(mu) + ' \n'
            ipFile.write(meanLine)

        ipFile.close()


    def getDataFromVideo(self, vid):
        data = []

        videoName = vid[0][0]
        vidObj = cv2.VideoCapture(videoName)
        bgsubVid = bgsub(vidObj)
        for labelData in vid:
            _, frameIdx, crops, points = labelData
            sy, by, sx, bx = crops
            cutOut = (bgsubVid[frameIdx])[sy: by + 1, sx: bx + 1]
            sizeY, sizeX = cutOut.shape[:2]
            points[0] = [points[0][0] * sizeX, points[0][1] * sizeY]
            points[1] = [points[1][0] * sizeX, points[1][1] * sizeY]
            # cv.imwrite('fish_'+ str(self.amountOfDataPutInCutOutData) +'.png', cutOut)
            # np.save('fish_'+ str(self.amountOfDataPutInCutOutData) +'.npy', np.array(points))
            data.append((cutOut, points))
        return data

    def getDataFromFolder(self, vid):
        folderName = vid[0][0]
        data = []

        bgsubVid = bgsubFolder(folderName)
        for labelData in vid:
            _, frameIdx, crops, points = labelData
            sy, by, sx, bx = crops
            cutOut = (bgsubVid[frameIdx])[sy: by + 1, sx: bx + 1]
            # cv.imwrite('temp.png', cutOut)
            # print('points: ', points)
            sizeY, sizeX = cutOut.shape[:2]
            points[0] = [points[0][0] * sizeX, points[0][1] * sizeY]
            points[1] = [points[1][0] * sizeX, points[1][1] * sizeY]

            data.append((cutOut, points))
        return data

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

    def __init__(self, cutout, bgsubInfo, *args, **kwargs):
        super(AnnotationsLabel, self).__init__(*args, **kwargs)
        self.markedPoints = None
        self.cutout = cutout
        self.bgsubInfo = bgsubInfo

if __name__ == '__main__':
    from Testing import *
    TestingWindow.testingClass = IntrinsicParametersPage
    run()








