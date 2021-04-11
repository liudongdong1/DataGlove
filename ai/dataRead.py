# -*-coding: utf-8 -*-
"""
    @Project: dataglove
    @File   : dataReader.py
    @Author : liudongodng1
    @E-mail : 3463264078@qq.com
    @Date   : 2021-04-11 18:45:06
"""

import numpy as np
from tqdm import tqdm
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


lableDigit={1: 'eight', 2: 'five', 3: 'four', 4: 'nine', 5: 'one', 6: 'seven', 7: 'six', 8: 'ten', 9: 'three', 10: 'two'}
lableWord={1: 'finger_heart', 2: 'gun', 3: 'horned_hand', 4: 'iloveu', 5: 'ok', 6: 'paper', 7: 'rock', 8: 'scissors', 9: 'thums_up', 10: 'well_played'}
 
class TorchDataset(Dataset):
    def __init__(self, basefolder, classtype):
        '''
        :param basefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\"
        :param type="digit" or "word" 有俩种情况的分类
        '''
        self.basefolder=os.path.join(basefolder,classtype)
        self.dataset=self.readData()
        if classtype=="digit":
            self.label_dic=dict(zip(lableDigit.values(), lableDigit.keys()))
        else:
            self.label_dic=dict(zip(lableWord.values(), lableWord.keys()))
        print("label:",self.label_dic)
        self.len=len(self.dataset)
 
    def __getitem__(self, i):
        index = i % self.len
        print(i,self.dataset[index][:-2],self.label_dic[self.dataset[index][-1]])
        return self.dataset[index][:-2],self.label_dic[self.dataset[index][-1]]    #data, label     
 
    def __len__(self):
        return len(self.dataset)
    
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
 
if __name__=='__main__':
    databasefoler=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex"
   
 
    epoch_num=2   #总样本循环次数
    batch_size=1  #训练时的一组数据的大小  
    train_data_nums=10
    max_iterate=int((train_data_nums+batch_size-1)/batch_size*epoch_num) #总迭代次数
 
    train_data = TorchDataset(basefolder=databasefoler, classtype="word")
    print(len(train_data))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    print(next(iter(train_loader)))
 
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

