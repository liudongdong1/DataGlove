# -*-coding: utf-8 -*-
"""
    @Project: dataglove
    @File   : FlexDataRead.py
    @Author : liudongodng1
    @E-mail : 3463264078@qq.com
    @Date   : 2021-04-11 18:45:06
"""
import matplotlib.pyplot as plt
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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from transformHandle import *
from util.transformHandle import *

def getlabel(folder):
    lable={}
    i=1
    for foldername in os.listdir(folder):
        lable[i]=foldername
        i=i+1
    print(lable)


lableDigit={0: 'one', 1: 'two', 2: 'three', 3:'four', 4:'five' , 5: 'six', 6: 'seven', 7: 'eight', 8:  'nine'}
lableWord={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
fingers = {"Little finger","Ring finger","Middle finger","Index finger","Thumb finger"}
class DataReadTrain(Dataset):

    def __init__(self,basefolder,classtype,transform):
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
        self.transform=transform
        self.data=self.readData()
        

    def readData(self):
        data=[]
        for tempfile in os.listdir(self.basefolder):
            filename=os.path.join(self.basefolder,tempfile)
            tempdata=np.loadtxt(filename,delimiter=',')
            label=self.label_dic[tempfile.split(".")[0]]
            label=np.full((len(tempdata),1),label,dtype=np.int)
            tempdata=np.column_stack((tempdata,label))
            #print("filename: {} label {} ".format(tempfile,label))
            if len(data)==0:
                data=tempdata
            else:
                data=np.row_stack((data,tempdata))
        return data

    
    def databoxplot(self,nrows,ncols,savefilename):
        descdata={}
        for i in range(0,len(self.data)):
            if not self.data[i][-1] in descdata:
                descdata[self.data[i][-1]]=[]
            descdata[self.data[i][-1]].append(self.data[i][:-1])
        print(descdata.keys())
        #np.savetxt("1.txt", descdata[1.0],fmt='%d',delimiter=',')
        i=0
        blotplist=[]
        fig,axes=plt.subplots(nrows,ncols,figsize=(100,30))
        print("train file:",self.label_dic.values())
        #for key in descdata.keys():

        for value,key in self.label_dic.items():
            descdata[key]=np.array(descdata[key]).T
            #print(type(descdata[key]),descdata[key].shape,descdata[key])
            bplot1=axes[int(i/ncols),i%ncols].boxplot(descdata[key].tolist(),
                    vert=True,
                    patch_artist=True,labels=fingers)
            axes[int(i/ncols),i%ncols].yaxis.grid(True) #在y轴上添加网格线
            ##axes[int(i/6),(i-int(i/6))%4].set_xticks(["A","B","C","D","E"] ) #指定x轴的轴刻度个数
            axes[int(i/ncols),i%ncols].set_xlabel(value) #设置x轴名称
            axes[int(i/ncols),i%ncols].set_ylabel('ylabel') #设置y轴名称
            blotplist.append(bplot1)
            i=i+1
        colors = ['pink', 'lightblue', 'lightgreen','red','orange']
        print("blot length:",len(blotplist))
        for bplot in blotplist:
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.savefig(savefilename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        '''
           return:
            train: 小拇指-》大拇指数据， 伸直表示0度， 
        '''
        oneRecord=self.data[item]
        #label=np.eye(len(self.label_dic))[int(oneRecord[-1]-1)]      #onehot 编码
        label=np.array(oneRecord[-1])
        train=oneRecord[:5]
        #print(type(train))
        if self.transform:
            train=self.transform(train)
        #print(train.shape,label.shape,"type",type(train),type(label))
        return torch.from_numpy(train).float(), torch.from_numpy(label).long()   #转化为tensor向量

class DataReadTest(Dataset):
    
    def __init__(self,basefolder,classtype,transform):
        '''
        :param basefolder=r"../../data/temp/picFlex/"
        :param type="digit" or "word" 有俩种情况的分类,   test是一个类别目录，然后该目录下有该类别的txt文件
        '''
        self.basefolder=basefolder
        if classtype[0]=='d':
            self.label_dic=dict(zip(lableDigit.values(), lableDigit.keys()))
        else:
            self.label_dic=dict(zip(lableWord.values(), lableWord.keys()))
        print("label:",self.label_dic)
        self.transform=transform
        self.data=self.readData()
        

    def readData(self):
        data=[]
        for labelfolder in os.listdir(self.basefolder):
            for tempfile in os.listdir(os.path.join(self.basefolder,labelfolder)):
                filename=os.path.join(self.basefolder,labelfolder,tempfile)
                tempdata=np.loadtxt(filename,delimiter=',')
                label=self.label_dic[labelfolder]
                print("filename: {} label {} ".format(filename,label))
                label=np.full((len(tempdata),1),label,dtype=np.int)
                tempdata=np.column_stack((tempdata,label))
                if len(data)==0:
                    data=tempdata
                else:
                    data=np.row_stack((data,tempdata))
        return data

    def databoxplot(self,nrows,ncols,savefilename):
        descdata={}
        for i in range(0,len(self.data)):
            if not self.data[i][-1] in descdata:
                descdata[self.data[i][-1]]=[]
            descdata[self.data[i][-1]].append(self.data[i][:-1])
        print(descdata.keys())
        #np.savetxt("1.txt", descdata[1.0],fmt='%d',delimiter=',')
        i=0
        blotplist=[]
        fig,axes=plt.subplots(nrows,ncols,figsize=(100,30))
        #for key in descdata.keys():
        print("labels:",self.label_dic.values())
        for value,key in self.label_dic.items():
            descdata[key]=np.array(descdata[key]).T
            #print(type(descdata[key]),descdata[key].shape,descdata[key])
            bplot1=axes[int(i/ncols),i%ncols].boxplot(descdata[key].tolist(),
                    vert=True,
                    patch_artist=True,labels=fingers)
            axes[int(i/ncols),i%ncols].yaxis.grid(True) #在y轴上添加网格线
            ##axes[int(i/6),(i-int(i/6))%4].set_xticks(["A","B","C","D","E"] ) #指定x轴的轴刻度个数
            axes[int(i/ncols),i%ncols].set_xlabel(value) #设置x轴名称
            axes[int(i/ncols),i%ncols].set_ylabel('ylabel') #设置y轴名称
            blotplist.append(bplot1)
            i=i+1
        colors = ['pink', 'lightblue', 'lightgreen','red','orange']
        print("blot length:",len(blotplist))
        for bplot in blotplist:
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.savefig(savefilename)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        '''
           return:
            train: 小拇指-》大拇指数据， 伸直表示0度， 
        '''
        oneRecord=self.data[item]
        #label=np.eye(len(self.label_dic))[int(oneRecord[-1]-1)]      #onehot 编码
        label=np.array(oneRecord[-1])
        train=oneRecord[:5]
        #print(type(train))
        if self.transform:
            train=self.transform(train)
        
        #print(train.shape,label.shape)
        return torch.from_numpy(train).float(), torch.from_numpy(label).long()   #转化为tensor向量


def FlexSensorDataReadTestFunction():
    dataset=DataReadTrain(r"../../../data/flexData/chars/char1","word",sixPointAdj)
    dataset.databoxplot(6,5,"/home/iot/jupyter/root_dir/liudongdong/src/ai/output/dataplot/char_train.png")
    print("train data:",dataset[0][0])
    print("test label:",dataset[0][1])
    dataset=DataReadTest(r"../../../data/flexData/chars/charFlex/26char","word",None)
    dataset.databoxplot(6,5,"/home/iot/jupyter/root_dir/liudongdong/src/ai/output/dataplot/char_test.png")
    print("train data:",dataset[0][0])
    print("test label:",dataset[0][1])
    #dataset=DataReadTest(r"../../../data/flexData/digit/digitFlex_7days","d",None)
    #dataset.databoxplot(2,5,"/home/iot/jupyter/root_dir/liudongdong/src/ai/output/dataplot/digit_test.png")
    #print("train data:",dataset[0][0])
    #print("test label:",dataset[0][1])
    #dataset=DataReadTrain(r"../../../data/flexData/digit/digit1","d",None)
    #dataset.databoxplot(2,5,"/home/iot/jupyter/root_dir/liudongdong/src/ai/output/dataplot/digit_train.png")
if __name__=='__main__':
    #FlexSensorDataReadTestFunction()
    dataset=DataReadTest(r"../../../data/flexData/digit/digitFlex_7days","d",None)
    for indexc,data_val in enumerate(DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)):
        inputs,labels = data_val
        print(inputs[0],labels[0])
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

