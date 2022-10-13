
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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn.functional import gelu as gelu
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from torch.utils.data import DataLoader
from util.train_util import train, trainlog
from util.transformHandle import fivePoint

from util.config import DigitConfig5Point
from util.FlexDataRead import DataReadTrain, DataReadTest
from model.MLPmixer import MLPMixer
from torch.utils.mobile_optimizer import optimize_for_mobile
if not os.path.exists(DigitConfig5Point.SAVEDIR):
    os.makedirs(DigitConfig5Point.SAVEDIR)

logfile = '%s/trainDigit.log' % DigitConfig5Point.SAVEDIR
trainlog(logfile)

data_set = {}
data_set['train']=DataReadTrain(r"../../../data/flexData/digit/digit1","d",fivePoint)
data_set['val']=DataReadTest(r"../../../data/flexData/digit/digitFlex_7days","d",fivePoint)
# data_set['test']=FlexSensorDataRead(r"../../data/temp/picFlex/digitGen/test","digit",fivePoint)

dataloader={}
dataloader['train'] = DataLoader(dataset=data_set['train'], batch_size=DigitConfig5Point.BATCHSIZE, shuffle=True)
dataloader['val'] = DataLoader(dataset=data_set['val'], batch_size=DigitConfig5Point.BATCHSIZE, shuffle=True)
#dataloader['test'] = DataLoader(dataset=data_set['test'], batch_size=DigitConfig5Point.BATCHSIZE, shuffle=True)



mlp_mixer = MLPMixer(in_channels=5, image_size=5, num_patch=5, num_classes=DigitConfig5Point.NUM_CLASS,
                     dim=128, depth=4, token_dim=256, channel_dim=1024)
parameters = filter(lambda p: p.requires_grad, mlp_mixer.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

mlp_mixer.load_state_dict(torch.load("./output/DigitConfig5Point/weights-8-3046-[0.9997].pth"))
#/home/iot/jupyter/root_dir/liudongdong/src/ai/digit/output/DigitConfig5Point/weights-8-3046-[0.9997].pth
mlp_mixer = mlp_mixer.cuda()


## 添加部分cqx,2021.11.29
#model = mlp_mixer.cpu()
#model.eval()
#example = torch.rand(1, 5, 5)
#traced_script_module = torch.jit.trace(model, example)
#optimized_traced_model = optimize_for_mobile(traced_script_module)
#optimized_traced_model._save_for_lite_interpreter("/home/iot/jupyter/root_dir/liudongdong/src/ai/digit/output/DigitConfig5Point/modeltoandroid_jit.pt")
## 添加部分cqx,2021.11.29

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
optimizer = optim.Adam(mlp_mixer.parameters(), lr=DigitConfig5Point.LR)

#scheduler = StepLR(optimizer, step_size=1, gamma=DigitConfig5Point.GAMA)
scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
training_curve_data = {}

# train(mlp_mixer,
#           epoch_num=DigitConfig5Point.EPOCHES,
#           start_epoch=DigitConfig5Point.START_EPOCHES,
#           optimizer=optimizer,
#           criterion=criterion,
#           exp_lr_scheduler=scheduler,
#           data_set=data_set,
#           data_loader=dataloader,
#           save_dir=DigitConfig5Point.SAVEDIR,
#           print_inter=DigitConfig5Point.PRINT_INTERVAL,
#           val_inter=DigitConfig5Point.SAVS_INTERVAL)


### 这里被注释了
#save_path = os.path.join(DigitConfig5Point.SAVEDIR,
#                        'finaldigitTrans4m5p.pth')
#torch.save(mlp_mixer.state_dict(), save_path)

#------------------------ 测试代码部分----------------------
outputs=[]
ALLpreds=[]
ALLlabels=[]
mlp_mixer.eval()

from torchsummary import summary
from torch.autograd import Variable
#summary(mlp_mixer, ( 5, 5)) #模型参数，输入数据的格式

for batch_cnt_val, data_val in enumerate(dataloader['val']):
    inputs,labels = data_val
    
    inputs = Variable(inputs.cuda())
    #print(inputs.shape)
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

cm = confusion_matrix(ALLlabels,ALLpreds)

#print(cm)


labels_name={'1','2','3','4','5','6','7','8','9'}
plot_confusion_matrix(cm, labels_name,"")
plt.savefig('/home/iot/jupyter/root_dir/liudongdong/src/ai/digit/output/confusion/{}.png'.format('mixer_4m5P'), format='png')


#nohup python flexmixer_digitTrans4m5p.py >>../output/digit/4m5p.out