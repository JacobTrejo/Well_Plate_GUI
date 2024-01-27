import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from pathlib import Path
from PyQt5.QtCore import Qt, QBasicTimer, QDate, QMimeData
import PyQt5.QtCore as QtCore
import random
import imageio
import functools
import time
import numpy as np

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


class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setStyleSheet('background: ' + blue + ';')
        buttonsStyleSheet =  \
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
                super(titleLabel, self).__init__( *args, **kwargs)
                self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            def resizeEvent(self, a0):
                font = self.font()
                font.setPixelSize( int( self.height() * .7))
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
        button.setStyleSheet( buttonsStyleSheet)
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
        grid.addWidget(wellsBtn,3,0)

        parmsBtn = QPushButton('Calculate Intrinsic Parameters')
        parmsBtn.setFixedWidth(300)
        parmsBtn.setStyleSheet(buttonsStyleSheet)
        grid.addWidget(parmsBtn,4,0)


        self.setGeometry(300,300,300,300)
        self.setLayout(grid)
        self.show()

    def predictionPressed(self):
        # ex = predictionWindow()

        self.ex2 = predictionWindow()
        self.ex2.show()
        # time.sleep(2)
        self.close()


# class WindowExample(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         # set the title of main window
#         self.setWindowTitle('Sidebar layout - www.luochang.ink')
#
#         # set the size of window
#         self.Width = 800
#         self.height = int(0.618 * self.Width)
#         self.resize(self.Width, self.height)
#
#         # add all widgets
#         self.btn_1 = QPushButton('1', self)
#         self.btn_2 = QPushButton('2', self)
#         self.btn_3 = QPushButton('3', self)
#         self.btn_4 = QPushButton('4', self)
#
#         self.btn_1.clicked.connect(self.button1)
#         self.btn_2.clicked.connect(self.button2)
#         self.btn_3.clicked.connect(self.button3)
#         self.btn_4.clicked.connect(self.button4)
#
#         # add tabs
#         self.tab1 = self.ui1()
#         self.tab2 = self.ui2()
#         self.tab3 = self.ui3()
#         self.tab4 = self.ui4()
#
#         self.initUI()
#
#     def initUI(self):
#         left_layout = QVBoxLayout()
#         left_layout.addWidget(self.btn_1)
#         left_layout.addWidget(self.btn_2)
#         left_layout.addWidget(self.btn_3)
#         left_layout.addWidget(self.btn_4)
#         left_layout.addStretch(5)
#         left_layout.setSpacing(20)
#         left_widget = QWidget()
#         left_widget.setLayout(left_layout)
#
#         self.right_widget = QTabWidget()
#         self.right_widget.tabBar().setObjectName("mainTab")
#
#         self.right_widget.addTab(self.tab1, 'First')
#         self.right_widget.addTab(self.tab2, 'Second')
#         self.right_widget.addTab(self.tab3, 'Third')
#         self.right_widget.addTab(self.tab4, 'Fourth')
#
#         self.right_widget.setCurrentIndex(0)
#         self.right_widget.setStyleSheet('''QTabBar::tab{width: 0; \
#             height: 0; margin: 0; padding: 0; border: none;}''')
#
#         main_layout = QHBoxLayout()
#
#         # main_layout.addWidget(left_widget)
#         # main_layout.addWidget(self.right_widget)
#         # main_layout.setStretch(0, 40)
#         # main_layout.setStretch(1, 200)
#
#         main_layout.addWidget(self.right_widget)
#         main_layout.addWidget(left_widget)
#         main_layout.setStretch(0, 200)
#         main_layout.setStretch(1, 40)
#
#
#         main_widget = QWidget()
#         main_widget.setLayout(main_layout)
#         self.setCentralWidget(main_widget)
#         self.show()
#     # -----------------
#     # buttons
#
#     def button1(self):
#         self.right_widget.setCurrentIndex(0)
#
#     def button2(self):
#         self.right_widget.setCurrentIndex(1)
#
#     def button3(self):
#         self.right_widget.setCurrentIndex(2)
#
#     def button4(self):
#         self.right_widget.setCurrentIndex(3)
#
#     # -----------------
#     # pages
#
#     def ui1(self):
#         main_layout = QVBoxLayout()
#         main_layout.addWidget(QLabel('page 1'))
#         main_layout.addStretch(5)
#         main = QWidget()
#         main.setLayout(main_layout)
#         return main
#
#     def ui2(self):
#         main_layout = QVBoxLayout()
#         main_layout.addWidget(QLabel('page 2'))
#         main_layout.addStretch(5)
#         main = QWidget()
#         main.setLayout(main_layout)
#         return main
#
#     def ui3(self):
#         main_layout = QVBoxLayout()
#         main_layout.addWidget(QLabel('page 3'))
#         main_layout.addStretch(5)
#         main = QWidget()
#         main.setLayout(main_layout)
#         return main
#
#     def ui4(self):
#         main_layout = QVBoxLayout()
#         main_layout.addWidget(QLabel('page 4'))
#         main_layout.addStretch(5)
#         main = QWidget()
#         main.setLayout(main_layout)
#         return main

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

from PyQt5.QtCore import QSize
from PyQt5.Qt import QPainter

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

class predictionWindow(QMainWindow):

    def __init__(self):
        super().__init__()

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
        self.setCentralWidget(centralWidget)
        self.show()
    #
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
        self.ex = Window()
        self.close()


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


class DrawingWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(DrawingWindow, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        # label = ClickableImageViewer(QPixmap('wellplate.png'))
        label = ClickableImageViewer(QPixmap('wellplate.png'))

        self.setCentralWidget(label)

        self.setGeometry(300,300, 450, 450)
        self.show()







app = QApplication(sys.argv)
ex = DrawingWindow()

# ex = predictionWindow()
# ex = Window()

sys.exit(app.exec())













