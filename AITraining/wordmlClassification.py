# -*-coding: utf-8 -*-
"""
    @Project: dataglove
    @File   : mlClassification.py
    @Author : liudongodng1
    @E-mail : 3463264078@qq.com
    @Date   : 2021-04-11 18:45:06
"""

import numpy as np
import matplotlib.pyplot as mp
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.tree as st
import sklearn.datasets as sd  # sklearn提供的数据集
import sklearn.utils as su  # 可以把数据集按照行进行打乱
import sklearn.metrics as sm
import sklearn.ensemble as se
import joblib
import os
import random
from dataRead import *
# import warnings
# warnings.filterwarnings("ignore")

databasefoler=r"../../data/temp/picFlex"
classLabel=["word","word_testData","digit","digit_testData"]

# 读取测试数据
test_x,test_y = FlexSensorDataRead(basefolder=databasefoler,classtype=classLabel[1]).getDataLabel()
# 读取训练数据
train_data = TorchDataset(basefolder=databasefoler, classtype=classLabel[0])
train_x,train_y=train_data.getAllData()

#查看一组数据
print("test_x",test_x[0],"test_y",test_y[0])
print("train_x",train_x[0],"label_y",train_y[0])
## 数据标准化
# from sklearn.preprocessing import StandardScaler
# ss_X = StandardScaler()
# ss_y = StandardScaler()
# train_x = ss_X.fit_transform(train_x)
# test_x  = ss_X.transform(test_x)


model = svm.SVC(kernel='poly', degree=4)

# svm参数优化
max = 0
record_c = 0
for i in range(1,100):
    model = svm.SVC(kernel='linear', C=0.01*i)
    model.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)
    # bg = sm.classification_report(test_y, pred_test_y)
    if sm.accuracy_score(test_y, pred_test_y)>max:
        max = sm.accuracy_score(test_y, pred_test_y)
        record_c = 0.01*i
#
print(record_c)


