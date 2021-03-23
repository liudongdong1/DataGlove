from PyQt5 import QtCore, QtGui, QtWidgets
 
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(579, 369)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = MatplotlibWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 40, 421, 271))
        self.widget.setObjectName("widget")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setGeometry(QtCore.QRect(460, 70, 93, 81))
        self.startButton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.startButton.setIcon(icon)
        self.startButton.setIconSize(QtCore.QSize(80, 80))
        self.startButton.setObjectName("startButton")
        self.endButton = QtWidgets.QPushButton(self.centralwidget)
        self.endButton.setGeometry(QtCore.QRect(460, 200, 93, 81))
        self.endButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("stop.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.endButton.setIcon(icon1)
        self.endButton.setIconSize(QtCore.QSize(80, 80))
        self.endButton.setObjectName("endButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 579, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
 
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
 
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Audio"))
 
from MatplotlibWidget import MatplotlibWidget
 
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())