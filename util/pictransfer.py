import cv2
import mediapipe as mp
import os
import time
import json
import re
import numpy as np
import math
from ast import literal_eval
class DataTransfer():
    def __init__(self):
        self.datafolder="../../data/Image/word"   # 带转化数据集目录
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.handsOp=self.initMediaHandle()
        self.handatatmpimg="../../data/temp/image/word/"  #存储 mediapipe annotation 目录
        self.handatatmpflex="../../data/temp/picFlex/word/"  #存储 弯曲转化数据


    def initMediaHandle(self):
        hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.85)
        return hands
    def closehands(self):
        self.handsOp.close()

    def handsingleImage(self,imagepath,label):
        '''
            function: using mediapipe to extract hand 3d points
            parameters: 
                imagepath: the full path of the image waiting to be handled
            output: 
                #存储的归一化的三维坐标点；
                originleftright: #用于存储关键点三维坐标,原始mediapipe 输出
                originleftright：#用于记录左手右手状态， 原始mediapipe输出
        '''
        image = cv2.flip(cv2.imread(imagepath), 1)
        # Convert the BGR image to RGB before processing.
        results = self.handsOp.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Print handedness and draw hand landmarks on the image.
        #print('handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            return
        #print(imagepath)
        #print(imagepath.split("\\"))
        #label=imagepath.split("\\")[-2]       # Todo 这里通过文件 分隔符方式可能会存在错误
        if not os.path.exists(self.handatatmpflex+label):
            os.makedirs(self.handatatmpflex+label)
        print(label, "file",self.handatatmpflex+label+"/originleftright")
        with open(self.handatatmpflex+label+"/origindatafile",'a') as f1:  #用于存储关键点三维坐标
            f1.write(str(results.multi_hand_landmarks)+'\n')
        with open(self.handatatmpflex+label+"/originleftright",'a') as f2:  #用于记录左手右手状态
            f2.write(str(results.multi_handedness) + '\n')
        # 保存显示图片 测试成功
        self.showhandedData(image,results,label)  

    def showhandedData(self,image,results,label):
        annotated_image = image.copy()
        #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("image", 400, 490);  #设置窗口大小
        tmpfolder=os.path.join(self.handatatmpimg,label)
        if not os.path.exists(tmpfolder):
            os.makedirs(tmpfolder)
        for hand_landmarks in results.multi_hand_landmarks:
            #savejson = json.dumps(hand_landmarks)
            #with open("../config/record.json","a") as f:
                # json.dump(hand_landmarks,f)
                #print("save finished",file, savejson)
            self.mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            cv2.imwrite(
                os.path.join(tmpfolder,str(len(os.listdir(tmpfolder)))+".png"), cv2.flip(annotated_image, 1))
            #cv2.imshow('image', cv2.flip(image, 1))
            #time.sleep(2)
            #cv2.waitKey(0)  #该函数等待任何键盘事件指定的毫秒。如果您在这段时间内按下任何键，程序将继续运行。如果0被传递，它将无限期地等待一次敲击键。
        #cv2.destroyAllWindows()

    def singleFolderHandle(self,folder,label):
        '''
            folder: 完整的单个图片目录  C:\project\ASL\ArduinoProject\VR-Glove\DashBoard\data\DatasetAlpha\A
        '''
        for idx, file in enumerate(os.listdir(folder)):
            self.handsingleImage(os.path.join(folder,file),label)
    
    def batchFolderHandle(self):
        '''
            function: #使用mediapipe 对图片数据进行处理
        '''
        for label in os.listdir(self.datafolder):
            print("handle folder:", label)
            self.singleFolderHandle(os.path.join(self.datafolder,label),label)
    def hand_record(self,filename_open, filename_record):  #是得到左右手的数据吗
        '''
            function: 根据记录的handness状态数据，记录是左手还是右手
            parameters:
                filename_open: mediapipe 存储的原始左右手信息
                filename_record: 提取出来是左手还是右手记录
        '''
        with open(filename_open, 'r') as f:
            f1 = f.read()
            rawdata = re.findall(r"[RL][ie][gf][ht][t]?", f1)  # todo 这一部分正则转化的数据是什么
        with open(filename_record, 'a') as l_final:
            l_final.write(str(rawdata))

    # 处理科学计数法，字符串形式的科学计数法转换为float型
    def fun(self,str_num):
        before_e = float(str_num.split('e')[0])
        sign = str_num.split('e')[1][:1]
        after_e = int(str_num.split('e')[1][1:])

        if sign == '+':
            float_num = before_e * math.pow(10, after_e)
        elif sign == '-':
            float_num = before_e * math.pow(10, -after_e)
        else:
            float_num = None
            print('error: unknown sign')
        return float_num

    def hand_coordinate(self,filename_open,filename_record):
        '''
            存储每一行是21个点，每一个点有三个坐标xyz 到filename_record文件中
        '''
        # 正则表达式匹配，找出所有的数，这里已经包括了科学计数法，不必担心
        with open(filename_open, 'r') as f:
            f1 = f.read()
            rawdata = re.findall(r"[+-]?[0-9]\.[0-9]+[eE]?[+-]?[0-9]+", f1)

        # 创建并打开一个文件，往后面追加内容，记载的是转化后的数据，每一行是21个点，每一个点有三个坐标xyz
        fl_final = open(filename_record, 'a')
        # finaldata = []
        laterdata = []
        mid = []
        i = 0
        j = 0
        for x in rawdata:
            if 'e' in x:
                x = self.fun(x)
            else:
                x = float(x)
            x = float(format(x*100, '.4f'))  # 保存四位小数   扩大了100
            mid.append(x)
            i = i + 1
            if i == 3:
                i = 0
                laterdata.append(mid)
                mid = []
                j = j + 1
            if j == 21:
                j = 0
                # print(laterdata)
                # finaldata.append(laterdata)
                fl_final.write(str(laterdata) + '\n')
                laterdata = []

    def formateHandData(self,file1,file2,file3,file4,file5,label):
        '''
            parameters:
                file1:origindatafile;
                file2:originleftright;
                file3:coordinate;     #    3D 坐标
                file4:listtestlr_f;   #记录时左手还是右手状态
                file5:allcoordinate;   # 3D 坐标，手的状态（左手还是右手），label(具体字符含义)
        '''
        self.hand_record(file2, file4)
        self.hand_coordinate(file1, file3)
        with open(file3, 'r') as fl1:
            f1 = fl1.readlines()
            final_coordata = []
            # 将转换后的列表从str转换回真正的list '[]' - []
            for x in f1:
                final_coordata.append(literal_eval(x))
        with open(file4, 'r') as fl2:
            f2 = fl2.readlines()
            final_lrdata = []
            # 将转换后的列表从str转换回真正的list '[]' - []
            for x in f2:
                final_lrdata = literal_eval(x)
            # print(len(final_lrdata))
        final_all = []
        for i in range(len(final_coordata)):
            final_coordata[i].append(final_lrdata[i])
            final_coordata[i].append(label)
            final_all.append(final_coordata[i])

        with open(file5, 'w') as fl3:   # Todo 这里是存储到一个文件还是一个label目录一个文件? 不影响最后结果  
            fl3.write(str(final_all))

    def transforAll3DData(self):
        '''
            functions: 利用media匹配处理成文本数据里，批处理提取格式化数据，坐标，左右手
            parameters: 
                basefolder: default r"..\..\data\temp\picFlex\", 其中该目录下面有有A-zlabel 文件，每一个label文件下面是几个txt文件：
                origindatafile: mediapipe提取的 手部坐标点数据
                originleftright: mediapipe 提取的左右手的状态
                coordinate ： 最终格式化 [x,y,z], 手部状态数据
                listtestlr_f ： 最终手部状态数据
                allcoordinate ： 最终格式化 [x,y,z], 手部状态数据

        '''
        labels=os.listdir(self.handatatmpflex)
        print("所有的label为： ",labels)
        for label in labels:
            print("handle folder coordinate:", label)
            file1=os.path.join(self.handatatmpflex,label,"origindatafile")
            file2=os.path.join(self.handatatmpflex,label,"originleftright")
            file3=os.path.join(self.handatatmpflex,label,"coordinate")
            file4=os.path.join(self.handatatmpflex,label,"listtestlr_f")
            file5=os.path.join(self.handatatmpflex,label,"allcoordinate")
            self.formateHandData(file1,file2,file3,file4,file5,label)

    
    

if __name__ == '__main__':          
     
        
    #dataTransfer().handsingleImage(filepath)
    handle=DataTransfer()
    filepath=r"../../data/Image/six/"  #C:\project\ASL\ArduinoProject\VR-Glove\DashBoard\data\Image
    # for file in os.listdir(filepath):
    #     tempfile=filepath+file
    #     handle.handsingleImage(tempfile)
    handle.batchFolderHandle()
    handle.transforAll3DData()