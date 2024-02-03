from Auxilary import *
# from VideoPlayer3 import Widget
from VideoPlayer5 import Widget
from PyQt5 import QtCore, QtGui

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
        if len(self.drawingItems) > 0:
            for item in self.drawingItems:
                self.widget._scene.removeItem(item)
            self.drawingItems = []
            return

        print('You pressed the grid label')
        grid = np.load(self.gridPath)
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


if __name__ == '__main__':
    from Testing import *
    TestingWindow.testingClass = PredictionPage
    run()



