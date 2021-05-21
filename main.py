import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QTimer,Qt ,QTime
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPainter, QColor, QPen,QFont,QPixmap
import keyboard                                    #for pressing keys
from util.imghelper import Camera
from util.flexhelper import FlexSensor
from util.uhand import *
from util.pictransfer import DataTransfer

import time
import pyqtgraph as pg
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView
import os
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.filterOp import MovAvg                # 这里同时导入俩个包
from tools.flexQuantify import toangle_curve,fitFlexDataHandle
import warnings
warnings.filterwarnings("ignore")
class HandBase():
    def __init__(self,length):
        #原始数据的窗口 大小设置为30，用于判断 手势是否静止
        self.Ajudge=[]
        self.judgelength=30
        #滑动平均的窗口
        self.A=[]  #小拇指数据
        self.B=[]
        self.C=[]
        self.D=[]
        self.E=[]   #大拇指数据
        self.AavgFilter= MovAvg(10)
        self.BavgFilter= MovAvg(10)
        self.CavgFilter= MovAvg(10)
        self.DavgFilter= MovAvg(10)
        self.EavgFilter= MovAvg(10)
        self.length=length

        self.status=1  #用于判断是否进行识别操作, 0:表示运动状态，1表示静止状态
        self.label=1  #用于存储识别的结果  
        self.threshold=1.0  # 用于区分静止还是用的方差
        

    def add(self,data,parameters,minList):
        '''
            data=[a,b,c,d,e], 如果不是该格式，则舍弃
        '''
        if len(data)!=5:
            return

        if len(parameters)>0 and len(minList)>0:
            data=toangle_curve(data,parameters,minList)

        if len(self.Ajudge)>self.judgelength:
            self.Ajudge.pop(0)
        self.Ajudge.append(data[0])

        #可以再这里面添加数据拟合算法  --滑动平均处理算法
        data[0]=self.AavgFilter.update(data[0])
        data[1]=self.BavgFilter.update(data[1])
        data[2]=self.CavgFilter.update(data[2])
        data[3]=self.DavgFilter.update(data[3])
        data[4]=self.EavgFilter.update(data[4])
        #todo arduino 上面sleep（50ms),  选择什么参数比较合适
        if len(self.A)>self.length:
            self.A.pop(0)
            self.B.pop(0)
            self.C.pop(0)
            self.D.pop(0)
            self.E.pop(0)
        self.A.append(data[0])
        self.B.append(data[1])
        self.C.append(data[2])
        self.D.append(data[3])
        self.E.append(data[4])
        #进行状态判断，是否进行识别操作
        self.statechange(self.getVar())
        return [data[0],data[1],data[2],data[3],data[4]]
        
    def statechange(self,transfer):
        '''
        ;function: 状态转化图，并在转化图中 有运动转向静止的时候进行识别，及状态 0--1》 进行识别
        ;parameters:
            transfer: 原始数据的方差
        '''
        if transfer < self.threshold:
            if self.status==0:
                self.status=1
                self.recogniseHandle()
            else:
                self.status=1
        else:
            if self.status==0:
                self.status=0
            else:
                self.status=1

    def recogniseHandle(self):
        #todo 
        pass

    def getVar(self):
        return np.var(self.Ajudge)

    def getMean(self):
        '''
            计算窗口的均值
        '''
        return np.mean(self.A),np.mean(self.B),np.mean(self.C),np.mean(self.D),np.mean(self.E)

    def getLength(self):
        '''
            获得当前窗口的大小
        '''
        return len(self.A)

    def saveData(self,filename):
        '''
            存储窗口数据到文件中
        '''
        strflex=",".join([str(i) for i in self.A])+"\n"+",".join([str(i) for i in self.B])+"\n"+",".join([str(i) for i in self.C])+"\n"+",".join([str(i) for i in self.D])+"\n"+",".join([str(i) for i in self.E])
        #todo 存储到文件中，还是数据库中？ 存储到文件中已经完成，是否需要存储到数据库中
        with open(filename, 'w') as f:
            f.write(strflex)
        print("saveData Ok",filename)
        
    def clear(self):
        #清空窗口数据
        self.A.clear()
        self.B.clear()
        self.C.clear()
        self.D.clear()
        self.E.clear()

class Dashboard(QMainWindow):
    def __init__(self):
        super(Dashboard, self).__init__()
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.FramelessWindowHint)
        self.imgbasefolder="../data/Image/"              #存储 图片数据根目录，目录下内容是 label/picture
        self.flexbasefolder="../data/flexSensor/"       #存储 传感器数据根目录，目录下内容是 label/txt
        self.runtempbasefolder="../data/temp/"
        self.handData=HandBase(180)                     #记录临时数据窗口大小
        self.updateTimeInterval=20                     #视频和传感器 数据更新 定时器时间
        self.port="com9"                               # 弯曲传感器端口号
        self.frequency=9600
        self.parameters=[]
        self.voltage180=[]
        self.voltage0=[]
        self.flexsensor=FlexSensor(self.port,self.frequency,self.updateTimeInterval)
        self.porthand="com7"
        self.uhandcontrol=SerialOp(self.porthand, 9600, 0.3)
        self.showPage()

    def showPage(self):
        #加载显示首页
        #self.__timer.stop()
        uic.loadUi('ui_files/show.ui',self)
        self.initshowSlot()
        
    
    def initshowSlot(self):
        '''
            首页逻辑，初始化各种控件事件；
        '''
        #只要控件触发器设置
        self.create.clicked.connect(self.createGesture)
        self.scan_sinlge.clicked.connect(self.recogniseGesture)
        self.scan_sen.clicked.connect(self.handControl)
        self.exp2.clicked.connect(self.calibration)
        self.exit_button.clicked.connect(self.quitApplication)
        self.transfer.clicked.connect(self.transferFromPic)

        self.create.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.exp2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
 
        #==========================显示 首页各种手势============
        movie = QtGui.QMovie("icons/dashAnimation.gif")
        self.label_2.setMovie(movie)
        self.label_2.setGeometry(10,170,750,421)
        movie.start()

        # LCDNumber 时间设置
        self.__ShowColon = True #是否显示时间如[12:07]中的冒号，用于冒号的闪烁
        self.__timer = QTimer(self) #新建一个定时器
        #关联timeout信号和showTime函数，每当定时器过了指定时间间隔，就会调用showTime函数
        self.__timer.timeout.connect(self.showTime)
        
        self.__timer.start(1000) #设置定时间隔为1000ms即1s，并启动定时器
        self.lcdNumber.setNumDigits(8)
    
    def showTime(self):
        '''
            LCD 时间控件显示
        '''
        time = QTime.currentTime() #获取当前时间
        time_text = time.toString(Qt.DefaultLocaleLongDate) #获取HH:MM:SS格式的时间，在中国获取后是这个格式，其他国家我不知道，如果有土豪愿意送我去外国旅行的话我就可以试一试
        
        #冒号闪烁
        if self.__ShowColon == True:
            self.__ShowColon = False
        else:
            time_text = time_text.replace(':',' ')
            self.__ShowColon = True
        self.lcdNumber.display(time_text) #显示时间


    def transferFromPic(self):
        '''
            使用mediapipe对数据进行批量化转化
        '''
        userReply = QMessageBox.question(self, 'DataTransferOption', "Are you sure you want to GenerateDataset?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if userReply == QMessageBox.Yes:
            #todo  图片文件如何存储
            handle=DataTransfer()
            handle.batchFolderHandle(r"../data/DatasetAlpha")
            handle.transforAll3DData(r"../data/temp/picFlex")
        print("transferFromPic  数据转化完成")

    def quitApplication(self):
        """shutsdown the GUI window along"""
        userReply = QMessageBox.question(self, 'Quit Application', "Are you sure you want to quit this app?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if userReply == QMessageBox.Yes:
            keyboard.press_and_release('alt+F4')
    
    def createGesture(self):
        """ Custom gesture generation module，可以选择的采集摄像头数据或者是 flex sensor 数据"""
        self.__timer.stop()
        uic.loadUi('ui_files/create_gesture.ui', self)
        self.initCreateGestureSlot()

    def initCreateGestureSlot(self):
        '''
            数据采集模块逻辑，初始化控件事件
        '''
        self.transfer.clicked.connect(self.showPage)
        self.create.clicked.connect(self.createGesture)
        self.scan_sinlge.clicked.connect(self.recogniseGesture)
        self.scan_sen.clicked.connect(self.handControl)
        self.exp2.clicked.connect(self.calibration)
        self.exit_button.clicked.connect(self.quitApplication)  

        self.create.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.exp2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.camera = Camera(self.updateTimeInterval)  # 视频控制器
        print("update_frame start")
        self.camera.timer.timeout.connect(self.update_frame)

        if not self.flexsensor:
            self.flexsensor=FlexSensor(self.port,self.frequency,self.updateTimeInterval)
        print("update flexdata")
        #self.flex.timer.timeout.connect(self.add)  flex 和camera使用一个触发事件

        #这里通过手动点击 进行存储，并使用 box空间进行有选择存储
        self.plainTextEdit.setPlaceholderText("Enter Gesture label") 
        self.pushButton_2.clicked.connect(self.startCollect)
        self.pushButton.clicked.connect(self.saveCollectData)
        self.pushButton_3.clicked.connect(self.stopCollections)
        
        # self.figure = plt.figure(self.centralwidget)
        # self.canvas = FigureCanvas(self.figure)
        # self.canvas.setGeometry(QtCore.QRect(480,160,611,361))
        # self.plot_()

        #==========================使用g.PlotWidget进行绘图显示=======================================
        self.graphWidget = pg.PlotWidget(self.centralwidget)
        self.graphWidget.setGeometry(QtCore.QRect(480,160,611,361))
        # 设置图表标题、颜色、字体大小
        self.graphWidget.setTitle("FlexVoltage",color='008080',size='12pt')
        # 显示表格线
        self.graphWidget.showGrid(x=True, y=True)
        # 设置上下左右的label
        # 第一个参数 只能是 'left', 'bottom', 'right', or 'top'
        #self.graphWidget.setLabel("left", "voltage")
        self.graphWidget.setLabel("bottom", "timestamp")
        self.graphWidget.setBackground("#fefefe")  # 背景色
        self.curve1 = self.graphWidget.plot( pen=pg.mkPen(color='r', width=5),name="Sensor 1") # 线条颜色
        self.curve2 = self.graphWidget.plot(pen=pg.mkPen(color='b', width=5)) # 线条颜色
        self.curve3 = self.graphWidget.plot( pen=pg.mkPen(color='y', width=5)) # 线条颜色
        self.curve4 = self.graphWidget.plot( pen=pg.mkPen(color='k', width=5)) # 线条颜色
        self.curve5 = self.graphWidget.plot( pen=pg.mkPen(color='m', width=5)) # 线条颜色
        # print("set graphWidget ok")
        

    def plotFlexData(self):
        '''
            更新curve中的数据，以折现方式显示
        '''
        index=range(0,self.handData.getLength())
        # plot data: x, y values
        self.curve1.setData(index,self.handData.A)
        self.curve2.setData(index,self.handData.B)
        self.curve3.setData(index,self.handData.C)
        self.curve4.setData(index,self.handData.D)
        self.curve5.setData(index,self.handData.E)
        

    def startCollect(self):
        '''
            点击开始采集button，打开摄像头和flex传感器
        '''
        print("you clicked start Button")
        #camera time start
        if(self.checkBox.checkState() ==Qt.Checked):
            print("image clicked")
            self.camera.start(0)
        if(self.checkBox_2.checkState() ==Qt.Checked):
            print("flex clicked")
            self.flexsensor.start()

    def saveCollectData(self):
        '''
            存储摄像头数据 和 一定时间窗口大小5个弯曲传感器数据序列
        '''
        print("you clicked saveCollectData Button")
        ges_name = self.plainTextEdit.toPlainText().strip()
        if not os.path.exists(self.imgbasefolder+ges_name):
            os.mkdir(self.imgbasefolder+ges_name)
        if not os.path.exists(self.flexbasefolder+ges_name):
            os.mkdir(self.flexbasefolder+ges_name)
        print(ges_name)
        timestr=str(time.time())
        if(self.checkBox.checkState() ==Qt.Checked):
            cv2.imwrite(self.imgbasefolder+ges_name+"/{}.png".format(timestr),self.camera.frame)
        if(self.checkBox_2.checkState() ==Qt.Checked):
            filename=self.flexbasefolder+ges_name+"/{}.txt".format(timestr)
            self.handData.saveData(filename)
            self.handData.clear()
        print("save tofile")
        
    def stopCollections(self):
        '''
            关闭摄像头和flex传感器
        '''
        if(self.checkBox.checkState() ==Qt.Checked):
            self.camera.stop()
        if(self.checkBox_2.checkState() ==Qt.Checked):
            self.flexsensor.stop()

    def update_frame(self):
        """
            通过定时器定时更新frame画面和弯曲传感器数据并在窗口显示
        """
        if(self.checkBox.checkState() ==Qt.Checked):
            frame = self.camera.frame
            if frame is None:
                return None
            height2, width2, channel2 = frame.shape
            step2 = channel2 * width2
            # create QImage from image
            qImg2 = QImage(frame.data, width2, height2, step2, QImage.Format_RGB888)
            # show image in img_label
            try:
                self.label_3.setPixmap(QPixmap.fromImage(qImg2))
            except:
                pass
        if(self.checkBox_2.checkState() ==Qt.Checked):
            self.handData.add(self.flexsensor.Read_Line(),self.parameters,self.voltage180)
            self.plotFlexData()
        

    def recogniseGesture(self):
        self.__timer.stop()
        uic.loadUi('ui_files/gesture.ui', self)
        self.initrecogniseGesture()
        #Todo
        pass


    def initrecogniseGesture(self):
        self.transfer.clicked.connect(self.showPage)
        self.create.clicked.connect(self.createGesture)
        self.scan_sinlge.clicked.connect(self.recogniseGesture)
        self.scan_sen.clicked.connect(self.handControl)
        self.exp2.clicked.connect(self.calibration)
        self.exit_button.clicked.connect(self.quitApplication)  #Todo 要不要考虑读取数据的时候进行切换

        self.create.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.exp2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def calibration(self):
        '''
            Calibration 页面进入函数
        '''
        self.__timer.stop()
        uic.loadUi('ui_files/calibration.ui', self)
        self.initCalibration()
 
    def initCalibration(self):
        '''
            Calibration 控件对应触发事件初始化
        '''
        self.transfer.clicked.connect(self.showPage)
        self.create.clicked.connect(self.createGesture)
        self.scan_sinlge.clicked.connect(self.recogniseGesture)
        self.scan_sen.clicked.connect(self.handControl)
        self.exp2.clicked.connect(self.calibration)
        self.exit_button.clicked.connect(self.quitApplication)  #Todo 要不要考虑读取数据的时候进行切换

        if not self.flexsensor:
            self.flexsensor=FlexSensor(self.port,self.frequency,self.updateTimeInterval)

        self.flexsensor.timer.timeout.connect(self.updateFlexData)

        #==========================使用g.PlotWidget进行绘图显示=======================================
        self.graphWidget = pg.PlotWidget(self.centralwidget)
        self.graphWidget.setGeometry(QtCore.QRect(0,160,751,361))
        # 设置图表标题、颜色、字体大小
        self.graphWidget.setTitle("FlexVoltage",color='008080',size='12pt')
        # 显示表格线
        self.graphWidget.showGrid(x=True, y=True)
        # 设置上下左右的label
        # 第一个参数 只能是 'left', 'bottom', 'right', or 'top'
        #self.graphWidget.setLabel("left", "voltage")
        self.graphWidget.setLabel("bottom", "timestamp")
        self.graphWidget.setBackground("#fefefe")  # 背景色
        self.curve1 = self.graphWidget.plot( pen=pg.mkPen(color='r', width=5),name="Sensor 1") # 线条颜色
        self.curve2 = self.graphWidget.plot(pen=pg.mkPen(color='b', width=5)) # 线条颜色
        self.curve3 = self.graphWidget.plot( pen=pg.mkPen(color='y', width=5)) # 线条颜色
        self.curve4 = self.graphWidget.plot( pen=pg.mkPen(color='k', width=5)) # 线条颜色
        self.curve5 = self.graphWidget.plot( pen=pg.mkPen(color='m', width=5)) # 线条颜色
        
        #矫正
        self.pushButton_2.clicked.connect(self.startFlexFlow)
        self.pushButton.clicked.connect(self.startCalibration)
        self.pushButton_3.clicked.connect(self.stopCalibration)

    def updateFlexData(self):
        '''
            更新滑动窗口传感器数据，并进行显示
        '''
        self.handData.add(self.flexsensor.Read_Line(),self.parameters,self.voltage180)
        self.plotFlexData()

    def startFlexFlow(self):
        ''' 启动弯曲传感器 '''
        print("startFlexFlow")
        self.flexsensor.start()

    def stopCalibration(self):
        ''' 停止弯曲传感器 '''
        print("stopFlexFlow")
        #self.parameters=[(-0.003240615177718527, -1.0797538449289, 254.9987190113072), (-0.003669435884198588, -0.9853892729356424, 256.36065524512117), (-0.0007539783376082677, -1.5886557393867298, 270.5017293916536), (-0.0026255933547313813, -1.0448742767289487, 232.7190051479214), (-0.004238738592416967, -0.5000499582619828, 201.4819249173945)]
        #self.voltage180=[237.9, 272.8, 243.0, 246.7, 253.6]
        #为了测试，省去了calibration    #todo
        #self.parameters,self.voltage180=fitFlexDataHandle("./validation.txt",self.voltage180,self.voltage0)
        self.flexsensor.stop()
    
    def startCalibration(self):   # Todo 书写这一部分逻辑代码
        self.parameters=[]
        self.voltage180=[] 
        # message 提示框显示击鼓步骤
        reply = QMessageBox.information(self, '标题','请将双手伸直',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
        if(reply==QMessageBox.Yes):
            self.voltage0=self.handData.getMean()
            print("伸直状态：",self.voltage0)
        reply = QMessageBox.information(self, '标题','请将双手弯曲180度',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
        if(reply==QMessageBox.Yes):
            self.voltage180=self.handData.getMean()
            print("伸直状态：",self.voltage180)
        reply = QMessageBox.information(self, '标题','请缓慢从伸直到最大弯曲1次',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
        if(reply==QMessageBox.Yes):
            self.handData.clear()
        print("开始缓慢弯曲")
        
        reply = QMessageBox.information(self, '标题','记录数据结束',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)

        self.handData.saveData("./validation.txt")
        self.handData.clear()
        reply = QMessageBox.information(self, '标题','开始进行初始化矫正',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
        #todo  具体怎么量化还等待进一步实验   已经完成，用多项式进行量化处理
        
        self.parameters,self.voltage180=fitFlexDataHandle("./validation.txt",self.voltage180,self.voltage0)
        print("self.parameters",self.parameters,"self.voltage180",self.voltage180)
        


    def handControl(self):
        self.__timer.stop()
        uic.loadUi('ui_files/hand_control.ui', self)
        self.inithandControl()

    def inithandControl(self):
        '''
            机器手控制模块逻辑，初始化控件事件
        '''
        self.transfer.clicked.connect(self.showPage)
        self.create.clicked.connect(self.createGesture)
        self.scan_sinlge.clicked.connect(self.recogniseGesture)
        self.scan_sen.clicked.connect(self.handControl)
        self.exp2.clicked.connect(self.calibration)
        self.exit_button.clicked.connect(self.quitApplication)  

        self.create.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.exp2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.camera = Camera(self.updateTimeInterval)  # 视频控制器
        print("update_frame start")
        self.camera.timer.timeout.connect(self.update_frameControl)

        if not self.flexsensor:
            self.flexsensor=FlexSensor(self.port,self.frequency,self.updateTimeInterval)
        if not self.uhandcontrol.ser.isOpen():
            self.uhandcontrol=SerialOp("COM6", 9600, 0.3)

        self.pushButton_2.clicked.connect(self.startControl)
        self.pushButton_3.clicked.connect(self.stopControl)

        #==========================使用g.PlotWidget进行绘图显示=======================================
        self.graphWidget = pg.PlotWidget(self.centralwidget)
        self.graphWidget.setGeometry(QtCore.QRect(480,160,611,361))
        # 设置图表标题、颜色、字体大小
        self.graphWidget.setTitle("FlexVoltage",color='008080',size='12pt')
        # 显示表格线
        self.graphWidget.showGrid(x=True, y=True)
        # 设置上下左右的label
        # 第一个参数 只能是 'left', 'bottom', 'right', or 'top'
        #self.graphWidget.setLabel("left", "voltage")
        self.graphWidget.setLabel("bottom", "timestamp")
        self.graphWidget.setBackground("#fefefe")  # 背景色
        self.curve1 = self.graphWidget.plot( pen=pg.mkPen(color='r', width=5),name="Sensor 1") # 线条颜色
        self.curve2 = self.graphWidget.plot(pen=pg.mkPen(color='b', width=5)) # 线条颜色
        self.curve3 = self.graphWidget.plot( pen=pg.mkPen(color='y', width=5)) # 线条颜色
        self.curve4 = self.graphWidget.plot( pen=pg.mkPen(color='k', width=5)) # 线条颜色
        self.curve5 = self.graphWidget.plot( pen=pg.mkPen(color='m', width=5)) # 线条颜色

    def startControl(self):
        print("you clicked start Button")
        #camera time start
        if(self.checkBox.checkState() ==Qt.Checked):
            print("image clicked")
            self.camera.start(0)
        if(self.checkBox_2.checkState() ==Qt.Checked):
            print("flex clicked")
            self.flexsensor.start()

    def stopControl(self):
        '''
            关闭摄像头和flex传感器
        '''
        if(self.checkBox.checkState() == Qt.Checked):
            self.camera.stop()
        if(self.checkBox_2.checkState() ==Qt.Checked):
            self.flexsensor.stop()
    
    def update_frameControl(self):
        """
            通过定时器定时更新frame画面和弯曲传感器数据并在窗口显示
        """
        if(self.checkBox.checkState() ==Qt.Checked):
            frame = self.camera.frame
            if frame is None:
                return None
            height2, width2, channel2 = frame.shape
            step2 = channel2 * width2
            # create QImage from image
            qImg2 = QImage(frame.data, width2, height2, step2, QImage.Format_RGB888)
            # show image in img_label
            try:
                self.label_3.setPixmap(QPixmap.fromImage(qImg2))
            except:
                pass
        if(self.checkBox_2.checkState() ==Qt.Checked):
            benddata=self.handData.add(self.flexsensor.Read_Line(),self.parameters,self.voltage180)
            self.uhandcontrol.datasend(benddata)
            self.plotFlexData()
    
    def recogniseGesture(self):
        self.__timer.stop()
        uic.loadUi('ui_files/gesture.ui', self)
        self.initStaticGesture()
        
    def initStaticGesture(self):
        self.transfer.clicked.connect(self.showPage)
        self.create.clicked.connect(self.createGesture)
        self.scan_sinlge.clicked.connect(self.recogniseGesture)
        self.scan_sen.clicked.connect(self.handControl)
        self.exp2.clicked.connect(self.calibration)
        self.exit_button.clicked.connect(self.quitApplication)  #Todo 要不要考虑读取数据的时候进行切换

        if not self.flexsensor:
            self.flexsensor=FlexSensor(self.port,9600,self.updateTimeInterval)
        else:
            print(" flexsensor ready")
        print("update flexdata")
        #Todo 在这里添加模型识别结果
        self.flexsensor.timer.timeout.connect(self.updateFlexData)

        self.pushButton_2.clicked.connect(self.startFlexFlow)
        self.pushButton_3.clicked.connect(self.stopCalibration)

        #==========================使用g.PlotWidget进行绘图显示=======================================
        self.graphWidget = pg.PlotWidget(self.centralwidget)
        self.graphWidget.setGeometry(QtCore.QRect(480,160,611,361))
        # 设置图表标题、颜色、字体大小
        self.graphWidget.setTitle("FlexVoltage",color='008080',size='12pt')
        # 显示表格线
        self.graphWidget.showGrid(x=True, y=True)
        # 设置上下左右的label
        # 第一个参数 只能是 'left', 'bottom', 'right', or 'top'
        #self.graphWidget.setLabel("left", "voltage")
        self.graphWidget.setLabel("bottom", "timestamp")
        self.graphWidget.setBackground("#fefefe")  # 背景色
        self.curve1 = self.graphWidget.plot( pen=pg.mkPen(color='r', width=5),name="Sensor 1") # 线条颜色
        self.curve2 = self.graphWidget.plot(pen=pg.mkPen(color='b', width=5)) # 线条颜色
        self.curve3 = self.graphWidget.plot( pen=pg.mkPen(color='y', width=5)) # 线条颜色
        self.curve4 = self.graphWidget.plot( pen=pg.mkPen(color='k', width=5)) # 线条颜色
        self.curve5 = self.graphWidget.plot( pen=pg.mkPen(color='m', width=5)) # 线条颜色

    @property
    def frame(self):
        return self.camera.frame


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = Dashboard()
    win.show()
    sys.exit(app.exec())

# class MyDesiger(QMainWindow, Ui_MainWindow):
#     def __init__(self, parent=None):
#         super(MyDesiger, self).__init__(parent)
#         self.setupUi(self)

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     ui = MyDesiger()
#     print("111")
#     ui.show()
#     sys.exit(app.exec_())