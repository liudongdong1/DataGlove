import json
import pickle
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/iot/jupyter/root_dir/liudongdong/src/ai/util')
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn.functional import gelu as gelu
from torch.utils.data import DataLoader

from util.train_util import train, trainlog

from util.config import *
from util.FlexDataRead import DataReadTest
from model.MLPmixer import MLPMixer
from util.transformHandle import *
if not os.path.exists(Config.SAVEDIR):
    os.makedirs(Config.SAVEDIR)
logfile = '%s/trainlog.log' % Config.SAVEDIR
trainlog(logfile)
data_set = {}
data_set['test']=DataReadTest(r"/home/iot/jupyter/root_dir/liudongdong/data/flexData/digit/digitFlex_7days","d",fivePoint)
dataloader={}
dataloader['test'] = DataLoader(dataset=data_set['test'], batch_size=1000, shuffle=True)
#修改下这里
mlp_mixer = MLPMixer(in_channels=5, image_size=5, num_patch=5, num_classes=DigitConfig5Point.NUM_CLASS,
                     dim=128, depth=4, token_dim=256, channel_dim=1024)
parameters = filter(lambda p: p.requires_grad, mlp_mixer.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)
mlp_mixer.load_state_dict(torch.load('/home/iot/jupyter/root_dir/liudongdong/src/ai/digit/output/DigitConfig5Point/weights-8-3046-[0.9997].pth'))
outputs=[]
ALLpreds=[]
ALLlabels=[]
for batch_cnt_val, data_val in enumerate(dataloader['test']):
    inputs,labels = data_val
    outputs=mlp_mixer(inputs)
    _, preds = torch.max(outputs, 1)
    preds=np.reshape(preds,(-1,1))
    labels=np.reshape(labels,(-1,1))
    if len(ALLpreds)==0:
        ALLpreds=preds
        ALLlabels=labels
        #print(ALLpreds.shape,ALLlabels.shape)
    else:
        #print(ALLpreds.shape,ALLlabels.shape,preds.shape,labels.shape)
        ALLpreds=np.row_stack((ALLpreds,preds))
        ALLlabels=np.row_stack((ALLlabels,labels))
        print(ALLlabels.shape,ALLpreds.shape)
        
print(ALLpreds.shape,ALLlabels.shape)
from sklearn.metrics import classification_report
bg =classification_report( ALLlabels,ALLpreds)
print('分类报告：', bg, sep='\n')
#绘制混淆矩阵
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')

cm = confusion_matrix(ALLlabels,ALLpreds)

#print(cm)


labels_name={'1','2','3','4','5','6','7','8','9'}
plot_confusion_matrix(cm, labels_name,"")
plt.savefig('/home/iot/jupyter/root_dir/liudongdong/src/ai/digit/output/confusion/{}.png'.format('sevendays'), format='png')
ALLpreds=np.column_stack((ALLlabels,ALLpreds))
np.savetxt("testingResult.txt", ALLpreds,fmt='%d',delimiter=',')