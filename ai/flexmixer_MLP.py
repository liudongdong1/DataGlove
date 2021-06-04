
import json
import pickle
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import gelu as gelu
from torch.utils.data import DataLoader
from util.train_util import train, trainlog

from util.config import Config
from util.FlexDataReadNN import FlexSensorDataRead
from model.NeuralNet import NeuralNet

Config.SAVEDIR="./output/weight1/"
if not os.path.exists(Config.SAVEDIR):
    os.makedirs(Config.SAVEDIR)

logfile = '%s/trainlogv1.log' % Config.SAVEDIR
trainlog(logfile)

data_set = {}
data_set['train']=FlexSensorDataRead(r"../../data/temp/picFlex/charGen/train","word")
data_set['val']=FlexSensorDataRead(r"../../data/temp/picFlex/charGen/valid","word")
data_set['test']=FlexSensorDataRead(r"../../data/temp/picFlex/charGen/test","word")

dataloader={}
dataloader['train'] = DataLoader(dataset=data_set['train'], batch_size=500, shuffle=True)
dataloader['val'] = DataLoader(dataset=data_set['val'], batch_size=500, shuffle=True)
dataloader['test'] = DataLoader(dataset=data_set['test'], batch_size=500, shuffle=True)



model = NeuralNet(input_size=25, hidden_size=1024,hidden_size2=512,num_classes=26,dropout = 0.01)
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

#model.load_state_dict(torch.load("./output/weight1/weights-9-4983-[0.7806].pth"))

model = model.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=Config.LR)
scheduler = StepLR(optimizer, step_size=1, gamma=Config.GAMA)

training_curve_data = {}

train(model,
          epoch_num=100,
          start_epoch=Config.START_EPOCHES,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=scheduler,
          data_set=data_set,
          data_loader=dataloader,
          save_dir=Config.SAVEDIR,
          print_inter=Config.PRINT_INTERVAL,
          val_inter=Config.SAVS_INTERVAL)

