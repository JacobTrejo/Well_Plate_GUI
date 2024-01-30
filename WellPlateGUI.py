from MenuPage import MenuPage
from PredictionsPage import PredictionPage
from DefineWellsPage import DefineWellsPage
from IntrinsicParametersPage import IntrinsicParametersPage
from Auxilary import *
class WellPlateGUI(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(WellPlateGUI, self).__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):

        self.myCentralWidget = QStackedWidget()

        # Creating the pages
        menuPage = MenuPage(self.myCentralWidget)
        predictionWindow = PredictionPage()
        defineWellsPage = DefineWellsPage()
        intrinsicParametersWindow = IntrinsicParametersPage()

        # Adding the windows to the pages to the stack
        self.myCentralWidget.addWidget( menuPage )
        self.myCentralWidget.addWidget( predictionWindow )
        self.myCentralWidget.addWidget( defineWellsPage )
        self.myCentralWidget.addWidget( intrinsicParametersWindow )

        self.setCentralWidget(self.myCentralWidget)

        # self.testingWidget = MenuPage()
        # self.testingWidget = PredictionWindow()
        #
        # self.setCentralWidget(self.testingWidget)

        self.setGeometry(300, 300, 600, 600)
        self.show()

    def backPressed(self):
        self.myCentralWidget.setCurrentIndex(0)

    def predictionsPressed(self):
        # print('You Pressed the parents')
        self.myCentralWidget.setCurrentIndex(1)

    def defineWellsPressed(self):
        self.myCentralWidget.setCurrentIndex(2)

    def intrinsicParametersPressed(self):
        self.myCentralWidget.setCurrentIndex(3)

app = QApplication(sys.argv)
ex = WellPlateGUI()
# ex = PredictionWindow()
sys.exit(app.exec_())







