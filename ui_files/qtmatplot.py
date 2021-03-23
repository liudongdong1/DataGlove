#coding:utf-8

# 导入matplotlib模块并使用Qt5Agg
import matplotlib
matplotlib.use('Qt5Agg')
# 使用 matplotlib中的FigureCanvas (在使用 Qt5 Backends中 FigureCanvas继承自QtWidgets.QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import *

class App(QtWidgets.QDialog):
    def __init__(self,parent=None):
        # 父类初始化方法
        super(App,self).__init__(parent)
        
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('PyQt5结合Matplotlib绘制函数图像')
        # 几个QWidgets
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.button_plot = QtWidgets.QPushButton("绘制函数图像")
        self.line = QLineEdit() # 输入函数
        # 连接事件
        self.button_plot.clicked.connect(self.plot_)
        
        # 设置布局
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.line)
        layout.addWidget(self.button_plot)
        self.setLayout(layout)

    # 连接的绘制的方法
    def plot_(self):
        AgeList = ['10', '21', '12', '14', '25']
        NameList = ['Tom', 'Jon', 'Alice', 'Mike', 'Mary']

        #将AgeList中的数据转化为int类型
        AgeList = list(map(int, AgeList))

        # 将x,y轴转化为矩阵式
        self.x = np.arange(len(NameList)) + 1
        self.y = np.array(AgeList)

        #tick_label后边跟x轴上的值，（可选选项：color后面跟柱型的颜色，width后边跟柱体的宽度）
        plt.bar(range(len(NameList)), AgeList, tick_label=NameList, color='green', width=0.5)

        # 在柱体上显示数据
        for a, b in zip(self.x, self.y):
            plt.text(a-1, b, '%d' % b, ha='center', va='bottom')

        #设置标题
        plt.title("Demo")
		
		#画图
        self.canvas.draw()
        # 保存画出来的图片
        plt.savefig('1.jpg')

# 运行程序
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = App()
    main_window.show()
    app.exec()
