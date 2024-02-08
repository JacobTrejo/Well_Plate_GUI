import sys
from GUI_Pages.Auxilary import *

"""
    This File is used to test out the pages quickly.
    Just set the testing class to the object that you want to test and call run
"""
class TestingWindow(QMainWindow):
    testingClass = None
    xc = 300
    yc = 300
    width = 800
    height = 650

    def __init__(self, *args, **kwargs):
        super(TestingWindow, self).__init__(*args, **kwargs)
        # self.testingWidget = MenuPage()
        self.initUI()

    def initUI(self):
        self.setGeometry(TestingWindow.xc, TestingWindow.yc, TestingWindow.width, TestingWindow.height)
        self.setCentralWidget(TestingWindow.testingClass())
        self.show()

def run():
    app = QApplication(sys.argv)
    ex = TestingWindow()
    sys.exit(app.exec_())


