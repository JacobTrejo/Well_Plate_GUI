from GUI_Pages.Auxilary import *


class DrawingWidget(QLabel):

    def __init__(self, *args, **kwargs):
        super(DrawingWidget, self).__init__(*args, **kwargs)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setPen(Qt.magenta)
        center = QtCore.QPoint(0, 0)
        qp.drawEllipse(center, 10, 10)

class Overlay(QWidget):

    def __init__(self, *args, **kwargs):
        super(Overlay, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):

        myLayout = QGridLayout()
        backGround = QLabel('Text')
        backGround.setStyleSheet('background: green')
        backGround.setAlignment(Qt.AlignCenter)
        # foreground = QWidget()
        foreground = DrawingWidget()
        myLayout.addWidget(backGround, 0, 0, 1, 1)
        myLayout.addWidget(foreground, 0, 0, 1, 1)

        self.setLayout(myLayout)

from Testing import *
TestingWindow.testingClass = Overlay
run()





