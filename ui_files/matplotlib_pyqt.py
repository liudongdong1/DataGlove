"""
Module implementing myMainWindow.
"""
import sys
from PyQt5.QtCore import pyqtSlot , QThread
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
 
from Ui_matplotlib_pyqt import Ui_MainWindow
import Recorder
 
class myMainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(myMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.widget.setVisible(False)  # 绘图区域初始化为不可见
    
    @pyqtSlot()
    def on_startButton_clicked(self):
        """
        Slot documentation goes here.
        """
        self.widget.setVisible(True)
        self.widget.startAudio()   # 触发MatplotlibWidget的startAudio函数
        
    @pyqtSlot()
    def on_endButton_clicked(self):
        """
        Slot documentation goes here.
        """
        self.widget.setVisible(False)
        self.widget.endAudio()      # 触发MatplotlibWidget的endAudio函数
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = myMainWindow()
    ui.show()
    sys.exit(app.exec_())