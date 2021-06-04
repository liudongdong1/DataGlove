
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
from util.train_util import train, trainlog

from util.config import Config
from util.FlexDataRead import FlexSensorDataRead
from model.MLPmixer import MLPMixer

if not os.path.exists(Config.SAVEDIR):
    os.makedirs(Config.SAVEDIR)

logfile = '%s/trainlog.log' % Config.SAVEDIR
trainlog(logfile)

data_set = {}
data_set['train']=FlexSensorDataRead(r"../../data/temp/picFlex/charGen/train","word")
data_set['val']=FlexSensorDataRead(r"../../data/temp/picFlex/charGen/valid","word")
data_set['test']=FlexSensorDataRead(r"../../data/temp/picFlex/charGen/test","word")

dataloader={}
dataloader['train'] = DataLoader(dataset=data_set['train'], batch_size=Config.BATCHSIZE, shuffle=True)
dataloader['val'] = DataLoader(dataset=data_set['val'], batch_size=Config.BATCHSIZE, shuffle=True)
dataloader['test'] = DataLoader(dataset=data_set['test'], batch_size=Config.BATCHSIZE, shuffle=True)



mlp_mixer = MLPMixer(in_channels=5, image_size=5, patch_size=16, num_classes=Config.NUM_CLASS,
                     dim=128, depth=4, token_dim=256, channel_dim=1024)
parameters = filter(lambda p: p.requires_grad, mlp_mixer.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

mlp_mixer.load_state_dict(torch.load("./output/weights-25-6225-[0.9400].pth"))

mlp_mixer = mlp_mixer.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
optimizer = optim.Adam(mlp_mixer.parameters(), lr=Config.LR)

#scheduler = StepLR(optimizer, step_size=1, gamma=Config.GAMA)
scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
training_curve_data = {}

train(mlp_mixer,
          epoch_num=Config.EPOCHES,
          start_epoch=Config.START_EPOCHES,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=scheduler,
          data_set=data_set,
          data_loader=dataloader,
          save_dir=Config.SAVEDIR,
          print_inter=Config.PRINT_INTERVAL,
          val_inter=Config.SAVS_INTERVAL)

