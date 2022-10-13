# -*-coding: utf-8 -*-
"""
    @Project: dataglove
    @File   : dataReader.py
    @Author : liudongodng1
    @E-mail : 3463264078@qq.com
    @Date   : 2021-04-11 18:45:06
"""

import numpy as np
import pandas as pd
# import torch.utils.data as Data
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
from ast import literal_eval
from sklearn.preprocessing import LabelBinarizer
import os
import random
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

def getlabel(folder):
    lable={}
    i=1
    for foldername in os.listdir(folder):
        lable[i]=foldername
        i=i+1
    print(lable)

def getFeature5to15(a):
    b = []
    for i in range(len(a)):
        #b.append(a[i])
        ## 如果不想转化成特征向量，就将下面两行注释掉
        for j in range(i):
            b.append(a[i]-a[j])
    return b

lableDigit={1: 'eight', 2: 'five', 3: 'four', 4: 'nine', 5: 'one', 6: 'seven', 7: 'six', 8: 'ten', 9: 'three', 10: 'two'}
lableWord={1: 'finger_heart', 2: 'gun', 3: 'horned_hand', 4: 'iloveu', 5: 'ok', 6: 'paper', 7: 'rock', 8: 'scissors', 9: 'thums_up', 10: 'well_played'}
 
class TorchDataset(Dataset):
    '''
    ;function: 读取 有图片转化的手指角度数据，其中0-4 索引分别指代：大拇指，二拇指，三拇指，...五拇指
    '''
    def __init__(self, basefolder, classtype):
        '''
        :param basefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\"
        :param type="digit" or "word" 有俩种情况的分类
        '''
        self.basefolder=os.path.join(basefolder,classtype)
        self.dataset=self.readData()
        if classtype=="digit" or classtype=="digit_testData":
            self.label_dic=dict(zip(lableDigit.values(), lableDigit.keys()))
        else:
            self.label_dic=dict(zip(lableWord.values(), lableWord.keys()))
        print("label:",self.label_dic)
        self.len=len(self.dataset)
 
    def __getitem__(self, i):
        '''
        ;function: 根据ID返回 对于的一条记录，feature,label
        ;parameters: 
            i: 表示索引；
        ;return 
        '''
        index = i % self.len
        #print(i,self.dataset[index][:-2],self.label_dic[self.dataset[index][-1]])
        #存储的时候角度为 np.pi-angle  -->,并且将手指与 弯曲传感器手指关系对应
        #return [(np.pi-i)*180/np.pi for i in reversed(self.dataset[index][:-2])],self.label_dic[self.dataset[index][-1]]    #data, label     
        return getFeature5to15([(np.pi-i)*180/np.pi for i in reversed(self.dataset[index][:-2])]),self.label_dic[self.dataset[index][-1]]
    
    def __len__(self):
        return len(self.dataset)
    
    def getAllData(self):
        train_x=[]
        label_y=[]
        for i in range(0,self.len):
            train,label=self.__getitem__(i)
            train_x.append(train)
            label_y.append(label)
        print("数据长度：",len(train_x),len(label_y))
        #return train_x,label_y
        return np.asarray(train_x),np.asarray(label_y)   #输出格式  one record: [2.9531 2.2778 0.8467 0.6175 0.1996] 1
    
    def readData(self):
        coordinateDataTrain=[]
        for folder in os.listdir(self.basefolder):
            tempfile=os.path.join(self.basefolder,folder,"angle")
            with open(tempfile) as fileOp:
                lines=fileOp.readlines()            #每个文件长度为1
                #print(len(lines))
                for line in lines:
                    final_lrdata = literal_eval(line)
                    #print("data",final_lrdata)
                    for i in range(0,len(final_lrdata)):
                        if(final_lrdata[i][-2]=='Right'):
                            #if(i==0):
                                #print(final_lrdata[i])
                            coordinateDataTrain.append(final_lrdata[i])
            fileOp.close()
        return coordinateDataTrain
 
    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data
 
class FlexSensorDataRead(Dataset):

    def __init__(self,basefolder,classtype):
        '''
        :param basefolder=r"../../data/temp/picFlex/"
        :param type="digit" or "word" 有俩种情况的分类
        '''
        self.basefolder=os.path.join(basefolder,classtype)
        if classtype=="digit" or classtype=="digit_testData":
            self.label_dic=dict(zip(lableDigit.values(), lableDigit.keys()))
        else:
            self.label_dic=dict(zip(lableWord.values(), lableWord.keys()))
        print("label:",self.label_dic)
        self.test_data,self.test_label=self.readData()
        self.len=len(self.test_data)

    def getDataLabel(self):
        return self.test_data,self.test_label

    def readData(self):
        data=[]
        for folder in os.listdir(self.basefolder):
            tempdata=self.readOneFolderFlexData(folder)
            if len(data)==0:
                data=tempdata
            else:
                data=np.row_stack((data,tempdata))
        #print(data[:-2],"lable:",data[-1])
        feature=[getFeature5to15(record) for record in data[:,:-1]]
        #return data[:,:-1],data[:,-1]
        return feature, data[:,-1]
    
    def readOneFolderFlexData(self,folder):
        '''
        ;function: 读取一个folder下面所有的数据
        ;parameters:
            folder: 一个目录，形如  gun... 是一个label名称
        '''
        data=[]
        for file in os.listdir(os.path.join(self.basefolder,folder)):
            label=self.label_dic[folder]
            tempdata=self.readOneFlexData(os.path.join(self.basefolder,folder,file),label)
            if len(data)==0:
                data=tempdata
            else:
                data=np.row_stack((data,tempdata))
        return data

    def readOneFlexData(self,filename,label):
        '''
        ;function: 读取五个传感器电压数据,一个文件
        ;parameters: 
            filename: 存储五个传感器电压数据文件
        '''
        dicFlex=[]
        print("readOneFlexData: filename",filename)
        with open(filename,'r') as fileOp:
            lines=fileOp.readlines()
            if(len(lines)!=5):
                print("readOneFlexData error, lines=",len(lines))
                return
            else:
                for line in lines:
                    dicFlex.append([float(i) for i in line.split(',')])
            fileOp.close()

        #dicFlex 为 (5,*)大小的列表，通过转化为array变量进行转置操作获得 （*，5）大小数据
        # 如果直接通过一步遍历处理是不是快点，这里  暂时不考虑了
        dicFlex=np.array(dicFlex)  #转化为 array 类型
        dicFlex=np.transpose(dicFlex)  #进行转置
        dicFlex=np.column_stack((dicFlex,np.array([label for i in range(0,len(dicFlex))])))

        return dicFlex

def TorchDatasetTestFunction():
    databasefoler=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex"
   
 
    epoch_num=2   #总样本循环次数
    batch_size=1  #训练时的一组数据的大小  
    train_data_nums=10
    max_iterate=int((train_data_nums+batch_size-1)/batch_size*epoch_num) #总迭代次数
 
    train_data = TorchDataset(basefolder=databasefoler, classtype="word")
    print(len(train_data))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    print(next(iter(train_loader)))
    train_x,label_y=train_data.getAllData()
    print("one record:",train_x[0],label_y[0])
def FlexSensorDataReadTestFunction():
    dataset=FlexSensorDataRead(r"../../data/temp/picFlex/","word_testData")
    #oneRecord=dataset.readOneFlexData(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\word_testData\finger_heart\1618286770.818512.txt")
    print("test data:",dataset.test_data[0])
    print("test label:",dataset.test_label[0])
if __name__=='__main__':
    FlexSensorDataReadTestFunction()
    # [1]使用epoch方法迭代，TorchDataset的参数repeat=1
    # for epoch in range(epoch_num):
    #     for batch_image, batch_label in train_loader:
    #         image=batch_image[0,:]
    #         image=image.numpy()#image=np.array(image)
    #         image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    #         image_processing.cv_show_image("image",image)
    #         print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))
    #         # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
 
    # '''
    # 下面两种方式，TorchDataset设置repeat=None可以实现无限循环，退出循环由max_iterate设定
    # '''
    # train_data = TorchDataset(filename=train_filename, image_dir=image_dir,repeat=None)
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    # # [2]第2种迭代方法
    # for step, (batch_image, batch_label) in enumerate(train_loader):
    #     image=batch_image[0,:]
    #     image=image.numpy()#image=np.array(image)
    #     image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    #     image_processing.cv_show_image("image",image)
    #     print("step:{},batch_image.shape:{},batch_label:{}".format(step,batch_image.shape,batch_label))
    #     # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
    #     if step>=max_iterate:
    #         break
    # [3]第3种迭代方法
    # for step in range(max_iterate):
    #     batch_image, batch_label=train_loader.__iter__().__next__()
    #     image=batch_image[0,:]
    #     image=image.numpy()#image=np.array(image)
    #     image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    #     image_processing.cv_show_image("image",image)
    #     print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))
    #     # batch_x, batch_y = Variable(batch_x), Variable(batch_y)

