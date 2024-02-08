import sys

from GUI_Pages.Auxilary import *

# class MenuPage(QWidget):
#
#     def __init__(self, *args, **kwargs):
#         super(MenuPage, self).__init__(*args, **kwargs)
#
#         self.initUI()
#
#     def initUI(self):
#         self.setStyleSheet('background: ' + blue + ';')
#         buttonsStyleSheet = \
#             '''
#             border: 4px solid ''' + whiteBlue + ''';
#             color: ''' + white + ''';
#             font-family: 'shanti';
#             font-size: 16px;
#             border-radius: 25px;
#             padding: 15px 0;
#             margin-top: 20px}
#             *:hover{
#                 background:  ''' + whiteBlue + '''
#             }
#             '''
#
#         grid = QGridLayout(self)
#
#         class titleLabel(QLabel):
#             def __init__(self, *args, **kwargs):
#                 super(titleLabel, self).__init__(*args, **kwargs)
#                 self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
#
#             def resizeEvent(self, a0):
#                 font = self.font()
#                 font.setPixelSize(int(self.height() * .7))
#                 self.setFont(font)
#                 # self.setAlignment(Qt.AlignmentFlag.AlignCenter)
#                 # self.adjustSize()
#                 # self.setAlignment(Qt.AlignmentFlag.AlignHCenter)
#
#         title = titleLabel('Title')
#
#         title = QLabel('Title')
#         title.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         title.setMaximumHeight(100)
#         # title.setBaseSize(title.width(), 100)
#         title.setMinimumSize(100, 50)
#         grid.addWidget(title, 0, 0)
#
#         button = QPushButton('Predict')
#         button.setStyleSheet(buttonsStyleSheet)
#         button.setFixedWidth(300)
#         button.clicked.connect(self.predictionPressed)
#         grid.addWidget(button, 1, 0)
#
#         ccBtn = QPushButton('Calculate Correlation Coefficients')
#         ccBtn.setFixedWidth(300)
#         ccBtn.setStyleSheet(buttonsStyleSheet)
#         grid.addWidget(ccBtn, 2, 0)
#
#         wellsBtn = QPushButton('Define Wells')
#         wellsBtn.setFixedWidth(300)
#         wellsBtn.setStyleSheet(buttonsStyleSheet)
#         wellsBtn.clicked.connect(self.defineWellsPressed)
#         grid.addWidget(wellsBtn, 3, 0)
#
#         parmsBtn = QPushButton('Calculate Intrinsic Parameters')
#         parmsBtn.setFixedWidth(300)
#         parmsBtn.clicked.connect(self.intrinsicParametersPressed)
#         parmsBtn.setStyleSheet(buttonsStyleSheet)
#         grid.addWidget(parmsBtn, 4, 0)
#
#         temp = QWidget()
#         temp.setLayout(grid)
#         tempLayout = QVBoxLayout()
#         tempLayout.setContentsMargins(0,0,0,0)
#         tempLayout.addWidget(temp)
#
#         self.setLayout(tempLayout)
#
#     def predictionPressed(self):
#         self.parent().parent().predictionsPressed()
#         # print('You pressed the predictions button')
#
#     def defineWellsPressed(self):
#         self.parent().parent().defineWellsPressed()
#
#     def intrinsicParametersPressed(self):
#         self.parent().parent().intrinsicParametersPressed()


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

        # grid = QGridLayout(self)
        buttonsLayout = QVBoxLayout(self)

        class titleLabel(QLabel):
            def __init__(self, *args, **kwargs):
                super(titleLabel, self).__init__(*args, **kwargs)
                self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

            def resizeEvent(self, a0):
                font = self.font()
                font.setPixelSize(int(self.height() * .7))
                self.setFont(font)
                self.setStyleSheet('color: ' + firstColorName)

                # self.setAlignment(Qt.AlignmentFlag.AlignCenter)
                # self.adjustSize()
                # self.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        mainLayout = QVBoxLayout()

        title = titleLabel('Well Plate Pose Estimation')
        title.adjustSize()

        mainLayout.addWidget(title)

        # title = QLabel('Title')

        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setMaximumHeight(100)
        # title.setBaseSize(title.width(), 100)
        title.setMinimumSize(300, 50)
        title.setMinimumWidth(800)


        # grid.addWidget(title, 0, 0)
        buttonsLayout.addWidget(title)

        button = QPushButton('Predict')
        button.setStyleSheet(buttonsStyleSheet)
        button.setFixedWidth(300)
        button.clicked.connect(self.predictionPressed)
        # grid.addWidget(button, 1, 0)
        buttonsLayout.addWidget(button)

        # ccBtn = QPushButton('Calculate Correlation Coefficients')
        # ccBtn.setFixedWidth(300)
        # ccBtn.setStyleSheet(buttonsStyleSheet)
        # # grid.addWidget(ccBtn, 2, 0)
        # buttonsLayout.addWidget(ccBtn)

        wellsBtn = QPushButton('Define Wells')
        wellsBtn.setFixedWidth(300)
        wellsBtn.setStyleSheet(buttonsStyleSheet)
        wellsBtn.clicked.connect(self.defineWellsPressed)
        # grid.addWidget(wellsBtn, 3, 0)
        buttonsLayout.addWidget(wellsBtn)

        parmsBtn = QPushButton('Calculate Intrinsic Parameters')
        parmsBtn.setFixedWidth(300)
        parmsBtn.clicked.connect(self.intrinsicParametersPressed)
        parmsBtn.setStyleSheet(buttonsStyleSheet)
        # grid.addWidget(parmsBtn, 4, 0)
        buttonsLayout.addWidget(parmsBtn)

        buttons = QWidget()
        buttons.setLayout(buttonsLayout)

        # The following is important to correctly center the object,
        # I feel like its due to aligment not doint their job
        outherWidgetForButtons = QWidget()
        outherWidgetForButtonsLayout = QHBoxLayout()

        leftSpacer = QWidget()
        rightSpacer= QWidget()

        outherWidgetForButtonsLayout.addWidget(leftSpacer, 1)
        outherWidgetForButtonsLayout.addWidget(buttons, 0)
        outherWidgetForButtonsLayout.addWidget(rightSpacer, 1)
        outherWidgetForButtons.setLayout(outherWidgetForButtonsLayout)

        mainLayout.addWidget(outherWidgetForButtons)

        # self.setStyleSheet('border: 1px solid')
        temp = QWidget()
        temp.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding )
        # temp.setLayout(buttonsLayout)
        temp.setLayout(mainLayout)

        tempLayout = QHBoxLayout()
        tempLayout.setContentsMargins(0,0,0,0)
        tempLayout.setSpacing(0)
        leftSpacer = QWidget()
        rightSpacer = QWidget()

        tempLayout.addWidget(leftSpacer, 1)
        tempLayout.addWidget(temp, 0)
        tempLayout.addWidget(rightSpacer, 1)

        self.setLayout(tempLayout)

    def predictionPressed(self):
        self.parent().parent().predictionsPressed()
        # print('You pressed the predictions button')

    def defineWellsPressed(self):
        self.parent().parent().defineWellsPressed()

    def intrinsicParametersPressed(self):
        self.parent().parent().intrinsicParametersPressed()

if __name__ == '__main__':
    from Testing import *
    TestingWindow.testingClass = MenuPage
    run()
