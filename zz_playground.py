import numpy as np
import imageio
import cv2 as cv
from Auxilary import *
from bgsub import bgsubFolder
import cv2 as cv

grid = np.load('grids/wellplate.npy')

bgsubVideo = bgsubFolder('videos/wellPlateImages')
frame0 = bgsubVideo[20]
frameShape = frame0.shape[:2]
grid *= frameShape[1]
grid = grid.astype(int)
amount = 0
for circle in grid:
    cx, cy, r = circle
    cutout = frame0[cy - r: cy + r + 1, cx - r: cx + r + 1]
    # Normalizing
    cutout = cutout.astype(float)
    cutout *= (255 / np.max(cutout))
    cv.imwrite('temp/cutout_' + str(amount) + '.png', cutout)
    amount += 1



# vid = Video()
# vid.setVideo('videos')
# print(vid.video)

# vid = cv.VideoCapture('videosActual/zebrafish.mp4')
# amountOfFrames = vid.get(cv.CAP_PROP_FRAME_COUNT)
# frame = vid.set(cv.CAP_PROP_POS_FRAMES, amountOfFrames - 1)
# ret, frame = vid.read()
# # cv.imwrite('temp.png', frame)
# print(frame.shape)

# class TestingWindow(QMainWindow):
#
#     def __init__(self, *args, **kwargs):
#         super(TestingWindow, self).__init__(*args, **kwargs)
#         self.initUI()
#
#     def initUI(self):
#         self.setGeometry(300,300, 600, 600)
#
#         colorWidget = QWidget()
#         colorWidgetLayout = QVBoxLayout()
#
#         firstColor = QWidget()
#         firstColor.setStyleSheet('background: red')
#         firstColor.setStyleSheet('background: ' + firstColorName)
#
#         secondColor = QWidget()
#         secondColor.setStyleSheet('background: yellow')
#         secondColor.setStyleSheet('background: ' + secondColorName)
#
#         thirdColor = QWidget()
#         thirdColor.setStyleSheet('background: green')
#         thirdColor.setStyleSheet('background: ' + thirdColorName)
#
#         fourthColor = QWidget()
#         fourthColor.setStyleSheet('background: blue')
#         fourthColor.setStyleSheet('background: ' + fourthColorName)
#
#         colorWidgetLayout.addWidget(firstColor)
#         colorWidgetLayout.addWidget(secondColor)
#         colorWidgetLayout.addWidget(thirdColor)
#         colorWidgetLayout.addWidget(fourthColor)
#
#         colorWidget.setLayout(colorWidgetLayout)
#
#         self.setCentralWidget(colorWidget)
#         self.show()
#
# app = QApplication(sys.argv)
# ex = TestingWindow()
# sys.exit(app.exec_())

