import numpy as np
import imageio
import cv2 as cv
from Auxilary import *



class TestingWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(TestingWindow, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        self.setGeometry(300,300, 600, 600)

        colorWidget = QWidget()
        colorWidgetLayout = QVBoxLayout()

        firstColor = QWidget()
        firstColor.setStyleSheet('background: red')
        firstColor.setStyleSheet('background: ' + firstColorName)

        secondColor = QWidget()
        secondColor.setStyleSheet('background: yellow')
        secondColor.setStyleSheet('background: ' + secondColorName)

        thirdColor = QWidget()
        thirdColor.setStyleSheet('background: green')
        thirdColor.setStyleSheet('background: ' + thirdColorName)

        fourthColor = QWidget()
        fourthColor.setStyleSheet('background: blue')
        fourthColor.setStyleSheet('background: ' + fourthColorName)

        colorWidgetLayout.addWidget(firstColor)
        colorWidgetLayout.addWidget(secondColor)
        colorWidgetLayout.addWidget(thirdColor)
        colorWidgetLayout.addWidget(fourthColor)

        colorWidget.setLayout(colorWidgetLayout)

        self.setCentralWidget(colorWidget)
        self.show()

app = QApplication(sys.argv)
ex = TestingWindow()
sys.exit(app.exec_())

