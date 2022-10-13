import json
import pickle
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn.functional import gelu as gelu
from torch.utils.data import DataLoader
from util.train_utilgnn import train, trainlog

from util.config import DigitConfigGCN
from util.FlexGNN import PictureDataRead,FlexSensorDataRead
from util.transformHandle import *
from model.GCN import GCN

if not os.path.exists(DigitConfigGCN.SAVEDIR):
    os.makedirs(DigitConfigGCN.SAVEDIR)

logfile = '%s/trainlog.log' % DigitConfigGCN.SAVEDIR
trainlog(logfile)

data_set = {}
data_set['train']=PictureDataRead(r"../../data/temp/picFlex/digit","digit",sixPointAdj)
data_set['val']=FlexSensorDataRead(r"../../data/temp/picFlex/digitGen/test","digit",sixPointAdj)
data_set['test']=FlexSensorDataRead(r"../../data/temp/picFlex/digitGen/test","digit",sixPointAdj)

dataloader={}
dataloader['train'] = DataLoader(dataset=data_set['train'], batch_size=DigitConfigGCN.BATCHSIZE, shuffle=True)
dataloader['val'] = DataLoader(dataset=data_set['val'], batch_size=DigitConfigGCN.BATCHSIZE, shuffle=True)
dataloader['test'] = DataLoader(dataset=data_set['test'], batch_size=DigitConfigGCN.BATCHSIZE, shuffle=True)

model = GCN(nfeat=1,
            nhid=180,
            nclass=10,
            dropout=0.001)
model.load_state_dict(torch.load("./output/digitGCN/weights-12-1899-[1.0000].pth"))

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=DigitConfigGCN.LR)

#scheduler = StepLR(optimizer, step_size=1, gamma=DigitConfigGCN.GAMA)
scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
training_curve_data = {}

model=model.cuda()
train(model,
          epoch_num=DigitConfigGCN.EPOCHES,
          start_epoch=DigitConfigGCN.START_EPOCHES,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=scheduler,
          data_set=data_set,
          data_loader=dataloader,
          save_dir=DigitConfigGCN.SAVEDIR,
          print_inter=DigitConfigGCN.PRINT_INTERVAL,
          val_inter=DigitConfigGCN.SAVS_INTERVAL)

step=0
numbers=0
for batch_cnt_val, data_val in enumerate(dataloader['test']):
    if step>3000:
        break
    step=step+1
    inputs,adj, labels = data_val
    inputs = Variable(inputs.cuda())
    adj = Variable(adj.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    outputs = model(inputs,adj)
    _, preds = torch.max(outputs, 1)
    print(preds,labels)
    batch_corrects = torch.sum((preds == labels)).item()
    numbers=numbers+batch_corrects
print("number:",numbers)