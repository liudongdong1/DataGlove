import sys
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QTimer,Qt ,QTime
from PyQt5.QtWidgets import QMessageBox
from functools import partial  
import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import keyboard                                    #for pressing keys
from util.imghelper import Camera
from util.flexhelper import FlexSensor
from util.faceUtil import Face_Recognizer
from util.uhand import *
from tools.model import CharRNN, MLPMixer
from tools.predict import *
from util.pictransfer import DataTransfer

from tools.config import Config
from tools.filterOp import MovAvg                # 这里同时导入俩个包
from tools.flexQuantify import toangle_curve,fitFlexDataHandle

import time
import pyqtgraph as pg

mlp_mixer = MLPMixer(in_channels=5, image_size=5, patch_size=16, num_classes=26,
                     dim=128, depth=4, token_dim=256, channel_dim=1024)
mlp_mixer.load_state_dict(torch.load(Config.MLPMIXER_WEIGHT,map_location='cpu'))

with open(Config.TEXT_CHARS, 'r', encoding='UTF-8') as f:
    text = f.read()
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encoded = np.array([char2int[ch] for ch in text])

charRNN = CharRNN(chars, 512, 2)
charRNN.load_state_dict(torch.load(Config.PRECHAR_WEIGHTS,map_location='cpu'))

lableWord={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
train_on_gpu=False
charclass=""
charpre=""


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
        self.threshold=10  # 用于区分静止还是用的方差

        self.recogeinseState=False

        self.charclassify=mlp_mixer
        self.charpredict=charRNN
        

    def add(self,data,parameters):
        '''
            data=[小拇指，四拇指，三拇指，二拇指，大拇指] 电压数据, 如果不是该格式，则舍弃
        '''
        if len(data)!=5:
            return

        # todo add transformer function
        if len(parameters)>0 :
            data=toangle_curve(data,parameters)   #[小拇指，四拇指，三拇指，二拇指，大拇指]  弯曲度，升直记作0度，弯曲记作180度。

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
        if self.recogeinseState:
            self.statechange(self.getVar(),data)
        return [data[0],data[1],data[2],data[3],data[4]]

    def setTrueRecogniseState(self):
        print("setTrueRecogniseState")
        self.recogeinseState=True

    def setFalseRecogniseState(self):
        print("setFalseRecogniseState")
        self.recogeinseState=False

    def statechange(self,transfer,data):
        '''
        ;function: 状态转化图，并在转化图中 有运动转向静止的时候进行识别，及状态 0--1》 进行识别
        ;parameters:
            transfer: 原始数据的方差
        '''
        if transfer < self.threshold:
            if self.status==0:
                self.status=1
                print("state:",self.status,"transfer:",transfer," self.threshold:",self.threshold,data)
                self.recogniseHandle(data)
            else:
                self.status=1
        else:
            if self.status==0:
                self.status=0
            else:
                self.status=0

    def recogniseHandle(self,data):
        print("recogniseHandle ")
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        data=(data - mu) / sigma
        temp=[]
        for i in range(0,len(data)):
            temp.append(data[i])
            for j in range(0,len(data)):
                if i!=j:
                    temp.append(data[i]-data[j])
        data=np.array(temp)
        data=np.reshape(data,(1,5,5))    #直接data.reshape() 不起作用
        data=torch.from_numpy(data).float()
        print("type:",type(data),data)
        output=self.charclassify(data)
        _, preds = torch.max(output, 1)
        index=preds.cpu().detach().numpy().tolist()[0]
        print("type(preds):",type(preds),"preds:",preds,index,"label:",lableWord[index])
        global charclass
        charclass=lableWord[index]
        global charpre
        charpre=sample(self.charpredict,size=2,prime=charclass,top_k=2)
        print("preds:{},charpre:{},charclassify:{}".format(preds,charpre,charclass))
        with open('result.txt',mode='a+') as f:
            f.write("preds:{},charpre:{},charclassify:{}".format(index,charclass,charpre))  # write 写入


    def getVar(self):
        vars=np.var(self.Ajudge)   #通过小拇指
        #print("长度为：",len(self.Ajudge),"方差位：",vars)
        return vars

    def getMean(self):
        '''
            计算窗口的均值
        '''
        return np.mean(self.A),np.mean(self.B),np.mean(self.C),np.mean(self.D),np.mean(self.E)

    def setLength(self,length):
        self.length=length

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
        self.imgbasefolder= Config.IMAGE_FOLDER         #存储 图片数据根目录，目录下内容是 label/picture
        self.flexbasefolder= Config.FLEX_FOLDER     #存储 传感器数据根目录，目录下内容是 label/txt
        self.runtempbasefolder=Config.RUNTEMP_BASEFOLDER
        self.handData=HandBase(Config.HANDLENGTH)                     #记录临时数据窗口大小
        self.updateTimeInterval=Config.UPTIME_INTERVAL                   #视频和传感器 数据更新 定时器时间
        self.port=Config.FLEX_PORT                             # 弯曲传感器端口号
        self.frequency=Config.FREQUENCY
        self.parameters=[]
        self.voltage180=[]
        self.voltage0=[]
        self.camera = Camera(self.updateTimeInterval)  # 视频控制器
        self.flexsensor=FlexSensor(self.port,self.frequency,self.updateTimeInterval)
        self.porthand=Config.UHAND_PORT
        self.uhandcontrol=SerialOp(self.porthand, self.frequency, Config.UHAND_CONNECTION_TIMEOUT)
        self.faceRec=Face_Recognizer(Config.FaceFolder)
        self.personName="None"
        #self.showPage()
        self.faceVerify()

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

    def faceVerify(self):
        """ Custom gesture generation module，可以选择的采集摄像头数据或者是 flex sensor 数据"""
        #self.__timer.stop()
        uic.loadUi('ui_files/FaceRec.ui', self)
        self.initfaceVerifySlot()

    def initfaceVerifySlot(self):
        '''
            人脸识别认证模块逻辑，初始化控件事件
        '''
        self.exit_button.clicked.connect(self.quitApplication)
        #这里通过手动点击 进行存储，并使用 box空间进行有选择存储
        self.lineEdit.setPlaceholderText(self.personName) 
        #断开信号槽
        #self.pushButton_3.disconnect()
        #self.pushButton_2.disconnect()
        #self.pushButton.disconnect()

        self.pushButton_3.clicked.connect(self.startCamera)
        self.pushButton_2.clicked.connect(self.faceVeridation)      # 身份认证
        self.pushButton.clicked.connect(self.faceRegister)   # 身份录入
        self.lineEdit.setPlaceholderText("")
    
    def startCamera(self):
        if not self.camera:
            self.camera = Camera(self.updateTimeInterval)  # 视频控制器
        print("update_frame start")
        
        self.camera.timer.timeout.connect(self.update_Faceframe)
        self.camera.start(0)

    def update_Faceframe(self):
        """
            通过定时器定时更新frame画面和弯曲传感器数据并在窗口显示
        """
        frame = self.camera.frame
        if frame is None:
            return None
        height2, width2, channel2 = frame.shape
        step2 = channel2 * width2
        # create QImage from image
        qImg2 = QImage(frame.data, width2, height2, step2, QImage.Format_RGB888)
        # show image in img_label
        try:
            self.label_2.setPixmap(QPixmap.fromImage(qImg2))
        except:
            pass

    def faceVeridation(self):
        try:
            #self.camera.stopTime()
            tempface=self.camera.frame
            print(type(tempface))
            if tempface  is None:
                print("图片数据为空，返回")
                self.camera.begin() 
                return
            cv2.imwrite("./{}.png".format(1),tempface)
            print("图片数据save")
            self.camera.pause()
            frame,name=self.faceRec._compareToDatabase(tempface)
            print("faceVeridation,name=",name)
            self.personName=name
            self.lineEdit.setPlaceholderText(self.personName) 
            height2, width2, channel2 = frame.shape
            step2 = channel2 * width2
            # create QImage from image
            qImg2 = QImage(frame.data, width2, height2, step2, QImage.Format_RGB888)
            # show image in img_label
            self.label_2.setPixmap(QPixmap.fromImage(qImg2))
        except Exception as e:
            print("faceVeridata error:",e)
            self.camera.begin() 
        reply = QMessageBox.information(self, '标题','身份认证成功，即将进入主页面',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
        if reply==QMessageBox.Yes:
            self.camera.timer.disconnect()
            self.pushButton_3.disconnect()
            self.pushButton_2.disconnect()
            self.pushButton.disconnect()
            self.camera.stop() 
            self.showPage()
        else:
            self.camera.begin()   
        

    def faceRegister(self):
        try:
            #self.camera.stopTime()
            facename=self.lineEdit.text().strip()
        
            if  facename!=None and len(facename)>1:
                tempface=self.camera.frame
                print(type(tempface))
                if tempface is None:
                    print("图片数据为空，返回")
                    self.camera.begin() 
                    return
                #cv2.imwrite("{}.png".format(2),tempface)
                if self.faceRec.faceRegister(tempface,facename):
                    print("人脸信息录入成功")
                else: print("人脸信息录入失败")
            else:
                reply = QMessageBox.information(self, '标题','请输入用户姓名',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            self.camera.begin()   
        except  Exception as e:
            print("faceRegister error:",e)
            self.camera.begin() 

        
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

        if not self.camera:
            self.camera = Camera(self.updateTimeInterval)  # 视频控制器
        print("initCreateGestureSlot start")

        #todo 
        #self.camera.timer.disconnect()

        self.camera.timer.timeout.connect(self.Create_update_frame)

        if not self.flexsensor:
            self.flexsensor=FlexSensor(self.port,self.frequency,self.updateTimeInterval)
        print("update flexdata")
        #self.flex.timer.timeout.connect(self.add)  flex 和camera使用一个触发事件

        #这里通过手动点击 进行存储，并使用 box空间进行有选择存储
        self.plainTextEdit.setPlaceholderText("Enter Gesture label") 

        #self.pushButton_2.disconnect()
        #self.pushButton_3.disconnect()
        #self.pushButton.disconnect()

        self.pushButton_2.clicked.connect(self.startCollect)
        self.pushButton.clicked.connect(self.saveCollectData)
        self.pushButton_3.clicked.connect(self.stopCollections)
        
        self.handData.clear()
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
        self.handData.setLength(Config.Create_Gesture_Length)
        #camera time start
        if(self.checkBox.checkState() ==Qt.Checked):
            print("image clicked")
            self.camera.start(0)
        if(self.checkBox_2.checkState() ==Qt.Checked):
            print("flex clicked")
            self.flexsensor.begin()
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
        try:
            self.camera.timer.disconnect()               ## 这里取消之前的 connection 机制
            self.flexsensor.timer.disconnect()
        except Exception as e:
            print(e)
    def Create_update_frame(self):
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
            self.handData.add(self.flexsensor.Read_Line(),self.parameters)
            self.plotFlexData()
        

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

        #self.flexsensor.timer.timeout.disconnect()

        self.flexsensor.timer.timeout.connect(self.CalibrationupdateFlexData)

        self.handData.clear()
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

        #self.pushButton_2.clicked.disconnect()
        #self.pushButton_3.clicked.disconnect()
        #self.pushButton.clicked.disconnect()

        self.pushButton_2.clicked.connect(self.startFlexFlow)
        self.pushButton.clicked.connect(self.startCalibration)
        self.pushButton_3.clicked.connect(self.stopCalibration)
    def CalibrationupdateFlexData(self,result=False):
        '''
            更新滑动窗口传感器数据，并进行显示
        '''
        self.handData.add(self.flexsensor.Read_Line(),self.parameters)
        self.plotFlexData()
        if result:
            self.label_2.setText("Result: {}  Predict words: {}".format(charclass,charpre))
            #print("Result: {}    Prewords: {}".format(charclass,charpre))


    def startFlexFlow(self):
        ''' 启动弯曲传感器 '''
        print("startFlexFlow")
        self.flexsensor.begin()
    def stopCalibration(self):
        ''' 停止弯曲传感器 '''
        print("stopFlexFlow")
        self.flexsensor.stop()
        #self.handData.setFalseRecogniseState()
    def startCalibration(self):   
        self.parameters=[]

        
        reply = QMessageBox.information(self, '标题','请缓慢从伸直到最大弯曲1次',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
        if(reply==QMessageBox.Yes):
            self.handData.clear()
        print("开始缓慢弯曲")
        reply = QMessageBox.information(self, '标题','记录数据结束',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
        ticks = time.time()
        temp=len(os.listdir(Config.ValidationFile))
        filename=Config.ValidationFile+"{}.txt".format(str(temp))
        self.handData.saveData(filename)
        self.handData.clear()
        reply = QMessageBox.information(self, '标题','开始进行初始化矫正',QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes) 
        
        filename=Config.ValidationFile+"{}.txt".format(str(2))
        self.parameters=fitFlexDataHandle(filename)
        print("self.parameters",self.parameters)
        


    def handControl(self):
        self.__timer.stop()
        uic.loadUi('ui_files/flex_control.ui', self)
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
        if not self.camera:
            self.camera = Camera(self.updateTimeInterval)  # 视频控制器
        print("update_frame start")
        #todo
        #self.camera.timer.disconnect()
        self.handData.clear()


        self.camera.timer.timeout.connect(self.update_frameHandControl)
        if not self.flexsensor:
            self.flexsensor=FlexSensor(self.port,self.frequency,self.updateTimeInterval)
        if not self.uhandcontrol:
            self.uhandcontrol=SerialOp(Config.UHAND_PORT, 9600, 0.3)

        #self.pushButton_2.disconnect()
        #self.pushButton_3.disconnect()
        #self.pushButton_4.disconnect()

        self.pushButton_2.clicked.connect(self.startControl)
        self.pushButton_3.clicked.connect(self.stopControl)
        self.pushButton_4.clicked.connect(self.mouseHandControl)
        
        #==========================使用g.PlotWidget进行绘图显示=======================================
        self.graphWidget = pg.PlotWidget(self.centralwidget)
        self.graphWidget.setGeometry(QtCore.QRect(480,160,611,361))
        self.graphWidget.setTitle("FlexVoltage",color='008080',size='12pt')
        self.graphWidget.showGrid(x=True, y=True)
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
            self.flexsensor.begin()
        

    def stopControl(self):
        '''
            关闭摄像头和flex传感器
        '''
        if(self.checkBox.checkState() == Qt.Checked):
            self.camera.stop()
        if(self.checkBox_2.checkState() ==Qt.Checked):
            self.flexsensor.stop()
        self.camera.timer.disconnect()

        

    def update_frameHandControl(self):
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
            benddata=self.handData.add(self.flexsensor.Read_Line(),self.parameters)
            self.uhandcontrol.datasend(benddata)
            self.plotFlexData()
    
    def mouseHandControl(self):
        print("you press mouseHandControl Button")
        self.camera.stop()
        self.flexsensor.stop()
        self.__timer.stop()
        uic.loadUi('ui_files/hand_control.ui', self)
        self.initMouseHandControl()


    def initMouseHandControl(self):
        '''
            鼠标控制机械手模块逻辑，初始化控件事件
        '''
        self.exit_button.clicked.connect(self.quitApplication)
        if not self.camera:
            self.camera = Camera(self.updateTimeInterval)  # 视频控制器
        print("update_Handframe start")
        try:
            self.camera.timer.disconnect()               ## 这里取消之前的 connection 机制
            self.flexsensor.timer.disconnect()
        except Exception as e:
            print(e)
        self.camera.timer.timeout.connect(self.update_Handframe)
        #self.camera.start(0)
        #

        

        if not self.uhandcontrol:
            self.uhandcontrol=SerialOp(Config.UHAND_PORT, 9600, 0.3)
        #self.pushButton_2.disconnect()
        #self.pushButton_3.disconnect()
        #self.slider1.disconnect()
        #self.slider1_2.disconnect()
        #self.slider1_3.disconnect()
        #self.slider1_4.disconnect()
        #self.slider1_5.disconnect()

        self.pushButton_2.clicked.connect(self.startHandControl)
        self.pushButton_3.clicked.connect(self.stopHandControl)
        self.slider1.valueChanged.connect(self.valuechange1)
        self.slider1_2.valueChanged.connect(self.valuechange2)
        self.slider1_3.valueChanged.connect(self.valuechange3)
        self.slider1_4.valueChanged.connect(self.valuechange4)
        self.slider1_5.valueChanged.connect(self.valuechange5)

    
    def valuechange1(self):
        #print('current slider value=%s'%self.slider1.value())
        size=self.slider1.value()
        self.uhandcontrol.datawriteSingleHand(5,size)
        self.label_5.setText('角度:{}'.format(size))   
    def valuechange2(self):
        #print('current slider value=%s'%self.slider1.value())
        size=self.slider1_2.value()
        self.uhandcontrol.datawriteSingleHand(4,size)
        self.label_7.setText('角度:{}'.format(size))   
    def valuechange3(self):
        #print('current slider value=%s'%self.slider1.value())
        size=self.slider1_3.value()
        self.uhandcontrol.datawriteSingleHand(3,size)
        self.label_8.setText('角度:{}'.format(size))   
    def valuechange4(self):
        #print('current slider value=%s'%self.slider1.value())
        size=self.slider1_4.value()
        self.uhandcontrol.datawriteSingleHand(2,size)
        self.label_10.setText('角度:{}'.format(size))   
    def valuechange5(self):
        #print('current slider value=%s'%self.slider1.value())
        size=self.slider1_5.value()
        self.uhandcontrol.datawriteSingleHand(1,size)
        self.label_13.setText('角度:{}'.format(size))   

    def stopHandControl(self):
        print("camera stop button clicked")
        self.camera.stop()
        self.flexsensor.stop()
        #进入到手势控制页面
        self.handControl()

    def startHandControl(self):
        print("camera start button clicked")
        self.camera.start(0)
        
    def update_Handframe(self):
        """
            通过定时器定时更新frame画面和弯曲传感器数据并在窗口显示
        """
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
        self.exit_button.clicked.connect(self.quitApplication) 

        self.create.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sen.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.scan_sinlge.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.exp2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        if not self.flexsensor:
            self.flexsensor=FlexSensor(self.port,9600,self.updateTimeInterval)
        else:
            print(" flexsensor ready")
        print("update flexdata")
        self.handData.setTrueRecogniseState()
        #self.flexsensor.timer.disconnect()
        #Todo 在这里添加模型识别结果
        self.flexsensor.timer.timeout.connect(partial(self.StaticGestureupdateFlexData,True))
        #self.pushButton_2.disconnect()
        #self.pushButton_3.disconnect()

        self.pushButton_2.clicked.connect(self.startFlexFlow)
        self.pushButton_3.clicked.connect(self.StaticStop)

        self.handData.clear()
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

    def StaticGestureupdateFlexData(self,result=False):
        '''
            更新滑动窗口传感器数据，并进行显示
        '''
        self.handData.add(self.flexsensor.Read_Line(),self.parameters)
        self.plotFlexData()
        if result:
            self.label_2.setText("Result: {}  Predict words: {}".format(charclass,charpre))
            #print("Result: {}    Prewords: {}".format(charclass,charpre))

    def StaticStop(self):
        ''' 停止弯曲传感器 '''
        print("stopFlexFlow")
        self.flexsensor.timer.disconnect()
        self.flexsensor.stop()
        self.handData.setFalseRecogniseState()

    @property
    def frame(self):
        return self.camera.frame


if __name__ == "__main__":
    # data=[26.558186204117863, 32.42994014489808, 18.37179090216694, 15.351224759483618, 103.23182514526965]
    # datahand=HandBase(180)
    # datahand.recogniseHandle(data)
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

