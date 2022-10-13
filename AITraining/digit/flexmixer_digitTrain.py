
import json
import pickle
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn.functional import gelu as gelu
from torch.utils.data import DataLoader


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.train_util import train, trainlog
from util.config import DigitConfig
from util.FlexDataRead import DataReadTrain, DataReadTest
from model.MLPmixer import MLPMixer

if not os.path.exists(DigitConfig.SAVEDIR):
    os.makedirs(DigitConfig.SAVEDIR)

logfile = '%s/trainDigit.log' % DigitConfig.SAVEDIR
trainlog(logfile)

data_set = {}
data_set['train']=DataReadTrain(r"../../../data/flexData/digit/digit1","d",None)
data_set['val']=DataReadTest(r"../../../data/flexData/digit/digitFlex_7days","d",None)

dataloader={}
dataloader['train'] = DataLoader(dataset=data_set['train'], batch_size=DigitConfig.BATCHSIZE, shuffle=True)
dataloader['val'] = DataLoader(dataset=data_set['val'], batch_size=DigitConfig.BATCHSIZE, shuffle=True)
#dataloader['test'] = DataLoader(dataset=data_set['test'], batch_size=DigitConfig.BATCHSIZE, shuffle=True)



mlp_mixer = MLPMixer(in_channels=5, image_size=5, patch_size=16, num_classes=DigitConfig.NUM_CLASS,
                     dim=128, depth=4, token_dim=256, channel_dim=1024)

# mlp_mixer = MLPMixer(in_channels=5, image_size=5, patch_size=16, num_classes=DigitConfig.NUM_CLASS,
#                      dim=128, depth=4, token_dim=256, channel_dim=1024)

parameters = filter(lambda p: p.requires_grad, mlp_mixer.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)


mlp_mixer = mlp_mixer.cuda()
mlp_mixer.load_state_dict(torch.load('./output/CharConfig5Point/weights-9-21927-[0.8056].pth'))

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
optimizer = optim.Adam(mlp_mixer.parameters(), lr=DigitConfig.LR)

#scheduler = StepLR(optimizer, step_size=1, gamma=DigitConfig.GAMA)
scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
training_curve_data = {}

# train(mlp_mixer,
#           epoch_num=DigitConfig.EPOCHES,
#           start_epoch=DigitConfig.START_EPOCHES,
#           optimizer=optimizer,
#           criterion=criterion,
#           exp_lr_scheduler=scheduler,
#           data_set=data_set,
#           data_loader=dataloader,
#           save_dir=DigitConfig.SAVEDIR,
#           print_inter=DigitConfig.PRINT_INTERVAL,
#           val_inter=DigitConfig.SAVS_INTERVAL)
outputs=[]
ALLpreds=[]
ALLlabels=[]
mlp_mixer.eval()

from torchsummary import summary
summary(mlp_mixer, ( 5, 5)) #模型参数，输入数据的格式

for batch_cnt_val, data_val in enumerate(dataloader['val']):
    inputs,labels = data_val
    
    inputs = Variable(inputs.cuda())
    
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

    outputs=mlp_mixer(inputs)
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

def plot_Matrix(cm, classes, title=None,  cmap=plt.cm.Blues):
    plt.rc('font',family='Times New Roman',size='8')   # 设置字体样式、大小
    
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) == 0:
                cm[i, j]=0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
                        ha="center", va="center",
                        color="white"  if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('./mixer_4m5pV2.png', dpi=300)
    plt.show()

cm = confusion_matrix(ALLlabels,ALLpreds)

#print(cm)
labels_name={'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}
plot_confusion_matrix(cm, labels_name,"")
plt.savefig('/home/iot/jupyter/root_dir/liudongdong/src/ai/char/output/confusion/{}.png'.format('mixer_4m5p'), format='png')
#nohup python mixer_4m5p.py >>../output/char/4m5p.out
