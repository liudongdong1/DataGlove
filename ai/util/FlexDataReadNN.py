# -*-coding: utf-8 -*-
"""
    @Project: dataglove
    @File   : FlexDataRead.py
    @Author : liudongodng1
    @E-mail : 3463264078@qq.com
    @Date   : 2021-04-11 18:45:06
"""

import numpy as np
import pandas as pd
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


lableDigit={1: 'eight', 2: 'five', 3: 'four', 4: 'nine', 5: 'one', 6: 'seven', 7: 'six', 8: 'ten', 9: 'three', 10: 'two'}
lableWord={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
 
class FlexSensorDataRead(Dataset):

    def __init__(self,basefolder,classtype):
        '''
        :param basefolder=r"../../data/temp/picFlex/"
        :param type="digit" or "word" 有俩种情况的分类
        '''
        self.basefolder=basefolder
        if classtype[0]=='d':
            self.label_dic=dict(zip(lableDigit.values(), lableDigit.keys()))
        else:
            self.label_dic=dict(zip(lableWord.values(), lableWord.keys()))
        print("label:",self.label_dic)
        self.data=self.readData()
        

    def readData(self):
        data=[]
        for tempfile in os.listdir(self.basefolder):
            filename=os.path.join(self.basefolder,tempfile)
            tempdata=np.loadtxt(filename,delimiter=',')
            label=self.label_dic[tempfile[0]]
            print("filename: {} label {} ".format(tempfile,label))
            label=np.full((len(tempdata),1),label,dtype=np.int)
            tempdata=np.column_stack((tempdata,label))
            if len(data)==0:
                data=tempdata
            else:
                data=np.row_stack((data,tempdata))
        return data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        oneRecord=self.data[item]
        #label=np.eye(len(self.label_dic))[int(oneRecord[-1]-1)]      #onehot 编码
        label=np.array(oneRecord[-1])
        train=self.standardization(oneRecord[:5])
        train=self.getFeature(train)
        return torch.from_numpy(train).float(), torch.from_numpy(label).long()   #转化为tensor向量

    def getFeature(self,data):
        temp=[]
        for i in range(0,len(data)):
            temp.append(data[i])
            for j in range(0,len(data)):
                if i!=j:
                    temp.append(data[i]-data[j])
        data=np.array(temp)
        data=np.reshape(data,(-1))    #直接data.reshape() 不起作用
        return data
            
    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
    
    def standardization(self,data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma


def FlexSensorDataReadTestFunction():
    dataset=FlexSensorDataRead(r"/home/iot/jupyter/root_dir/liudongdong/data/temp/picFlex/charGen/test","word")
    print("train data:",dataset[0][0])
    print("test label:",dataset[0][1])
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

