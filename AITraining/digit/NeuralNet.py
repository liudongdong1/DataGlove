
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import gelu as gelu
from torch.utils.data import DataLoader
from util.train_util import train, trainlog

from util.config import DigitNLConfig
from util.FlexDataRead import DataReadTrain, DataReadTest
from model.NeuralNet import NeuralNet
from util.transformHandle import *
DigitNLConfig.SAVEDIR="./output/weight1/"
if not os.path.exists(DigitNLConfig.SAVEDIR):
    os.makedirs(DigitNLConfig.SAVEDIR)

logfile = '%s/trainlogv1.log' % DigitNLConfig.SAVEDIR
trainlog(logfile)

data_set = {}
data_set['train']=DataReadTrain(r"../../../data/flexData/digit/digit1","d",None)
data_set['val']=DataReadTest(r"../../../data/flexData/digit/digitFlex_7days","d",None)

dataloader={}
dataloader['train'] = DataLoader(dataset=data_set['train'], batch_size=500, shuffle=True)
dataloader['val'] = DataLoader(dataset=data_set['val'], batch_size=500, shuffle=True)




model = NeuralNet(input_size=5, hidden_size=1024,hidden_size2=512,num_classes=10,dropout = 0.01)
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

#model.load_state_dict(torch.load("./output/weight1/weights-9-4983-[0.7806].pth"))

model = model.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=DigitNLConfig.LR)
scheduler = StepLR(optimizer, step_size=1, gamma=DigitNLConfig.GAMA)

training_curve_data = {}

train(model,
          epoch_num=100,
          start_epoch=DigitNLConfig.START_EPOCHES,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=scheduler,
          data_set=data_set,
          data_loader=dataloader,
          save_dir=DigitNLConfig.SAVEDIR,
          print_inter=DigitNLConfig.PRINT_INTERVAL,
          val_inter=DigitNLConfig.SAVS_INTERVAL)

outputs=[]
ALLpreds=[]
ALLlabels=[]
model.eval()

from torchsummary import summary
from torch.autograd import Variable
#summary(mlp_mixer, ( 5, 5)) #模型参数，输入数据的格式

for batch_cnt_val, data_val in enumerate(dataloader['val']):
    inputs,labels = data_val
    
    inputs = Variable(inputs.cuda())
    
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

    outputs=model(inputs)
    _, preds = torch.max(outputs, 1)
    preds=np.reshape(preds.cpu(),(-1,1))
    labels=np.reshape(labels.cpu(),(-1,1))
    if len(ALLpreds)==0:
        ALLpreds=preds
        ALLlabels=labels
        #print(ALLpreds.shape,ALLlabels.shape)
    else:
        #print(ALLpreds.shape,ALLlabels.shape,preds.shape,labels.shape)
        ALLpreds=np.row_stack((ALLpreds,preds))
        ALLlabels=np.row_stack((ALLlabels,labels))
        #print(ALLlabels.shape,ALLpreds.shape)

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
plt.savefig('/home/iot/jupyter/root_dir/liudongdong/src/ai/digit/output/confusion/{}.png'.format('NeuralNet'), format='png')

#nohup python NeuralNet.py >../output/digit/neuralnet.out