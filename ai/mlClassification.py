# -*-coding: utf-8 -*-
"""
    @Project: dataglove
    @File   : mlClassification.py
    @Author : liudongodng1
    @E-mail : 3463264078@qq.com
    @Date   : 2021-04-11 18:45:06
"""

import numpy as np
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

# 数据标准化
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
train_x = ss_X.fit_transform(train_x)
test_x  = ss_X.transform(test_x)


# model = svm.SVC(kernel='poly', degree=4)
## svm参数优化
# max = 0
# record_c = 0
# for i in range(1,100):
#     model = svm.SVC(kernel='linear', C=0.01*i)
#     model.fit(train_x, train_y)
#     pred_test_y = model.predict(test_x)
#     # bg = sm.classification_report(test_y, pred_test_y)
#     if sm.accuracy_score(test_y, pred_test_y)>max:
#         max = sm.accuracy_score(test_y, pred_test_y)
#         record_c = 0.01*i
# #
# print(record_c)
# # model = svm.SVC(kernel='poly',  random_state=0, gamma=0.078,degree=3)
model = svm.SVC(kernel='linear', C=0.58)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
#
# #
# # file = r'D:\pythonProject3\venv\svm_no_yes.joblib'
# # # 保存模型
# # joblib.dump(model,file)
# # # # 读取模型
# # svm_model = joblib.load(file)
# pred_test_y = svm_model.predict(test_x)
print(sm.accuracy_score(test_y, pred_test_y))
bg = sm.classification_report(test_y, pred_test_y)
print('分类报告：', bg, sep='\n')


##保存混淆矩阵
# def confusion_matrix(pred_test_y):
#     coonfusion_plot = [[0]*10 for i in range(10)]
#     for i in range(10):
#         for j in range(30):
#             coonfusion_plot[i][pred_test_y[i*30+j]] += 1
#     return coonfusion_plot

# pred_test = confusion_matrix(pred_test_y)
# f_confusion = r'D:\pythonProject3\venv\Scripts\confusion_matrix_no_yes.txt'
# with open(f_confusion, 'w') as fl3:
#     fl3.write(str(pred_test))
#


# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma":[1, 0.1, 0.01]}, cv=4)
# grid.fit(X, y)
# print("The best parameters are %s with a score of %0.2f" %(grid.best_params_, grid.best_score_))


#
# # # 【1】 KNN Classifier
# # # k-近邻分类器
# from sklearn.neighbors import KNeighborsClassifier
#
# knn = KNeighborsClassifier()
# knn.fit(train_x, train_y)
# pred_test_knn = knn.predict(test_x)
# print('knn:',format(sm.accuracy_score(test_y, pred_test_knn),'.4f'))
# print('recall:',format(sm.recall_score(test_y, pred_test_knn,average='macro'),'.4f'))
# print('F1:',format(sm.f1_score(test_y, pred_test_knn, average='macro'),'.4f'))
#
# # # # 【2】 Logistic Regression Classifier
# # # # 逻辑回归分类器
# # from sklearn.linear_model import LogisticRegression
# #
# # lg = LogisticRegression(penalty='l2')
# # lg.fit(train_x, train_y)
# # pred_test_lg = lg.predict(test_x)
# # print('accuracylg:',format(sm.accuracy_score(test_y, pred_test_lg),'.4f'))
#
# #
# # # 【3】 Random Forest Classifier
# # # 随机森林分类器
# from sklearn.ensemble import RandomForestClassifier
# #
# RFC = RandomForestClassifier(n_estimators=8)
# RFC.fit(train_x, train_y)
# pred_test_rfc = RFC.predict(test_x)
# print('accuracyrfc:',format(sm.accuracy_score(test_y, pred_test_rfc),'.4f'))
# print('recall:',format(sm.recall_score(test_y, pred_test_rfc,average='macro'),'.4f'))
# print('F1:',format(sm.f1_score(test_y, pred_test_rfc, average='macro'),'.4f'))
#
# #
# # # 【4】 Decision Tree Classifier
# # # 决策树分类器
# from sklearn import tree
#
# tre = tree.DecisionTreeClassifier()
# tre.fit(train_x, train_y)
# pred_test_tre = tre.predict(test_x)
# print('accuracytre:',format(sm.accuracy_score(test_y, pred_test_tre),'.4f'))
# print('recall:',format(sm.recall_score(test_y, pred_test_tre,average='macro'),'.4f'))
# print('F1:',format(sm.f1_score(test_y, pred_test_tre, average='macro'),'.4f'))


# # # # 【5】 SVM Classifier
## 支持向量机
# from sklearn.svm import SVC
#
# SV1 = svm.SVC(kernel='poly')
# SV2 = svm.SVC(kernel='rbf')
# SV3 = svm.SVC(kernel='linear')
# SV4 = svm.SVC(kernel='sigmoid')
# SV1.fit(train_x, train_y)
# SV2.fit(train_x, train_y)
# SV3.fit(train_x, train_y)
# SV4.fit(train_x, train_y)
# pred_test_sv1 = SV1.predict(test_x)
# pred_test_sv2 = SV2.predict(test_x)
# pred_test_sv3 = SV3.predict(test_x)
# pred_test_sv4 = SV4.predict(test_x)
# print('accuracySVM1poly:',format(sm.accuracy_score(test_y, pred_test_sv1),'.4f'))
# print('recall:',format(sm.recall_score(test_y, pred_test_sv1,average='macro'),'.4f'))
# print('F1:',format(sm.f1_score(test_y, pred_test_sv1, average='macro'),'.4f'))
#
# print('accuracySVM2rbf:',format(sm.accuracy_score(test_y, pred_test_sv2),'.4f'))
# print('recall:',format(sm.recall_score(test_y, pred_test_sv2,average='macro'),'.4f'))
# print('F1:',format(sm.f1_score(test_y, pred_test_sv2, average='macro'),'.4f'))
#
# print('accuracySVM3linear:',format(sm.accuracy_score(test_y, pred_test_sv3),'.4f'))
# print('recall:',format(sm.recall_score(test_y,pred_test_sv3,average='macro'),'.4f'))
# print('F1:',format(sm.f1_score(test_y, pred_test_sv3, average='macro'),'.4f'))
#
# print('accuracySVM4sigmoid:',format(sm.accuracy_score(test_y, pred_test_sv4),'.4f'))
# print('recall:',format(sm.recall_score(test_y, pred_test_sv4,average='macro'),'.4f'))
# print('F1:',format(sm.f1_score(test_y, pred_test_sv4, average='macro'),'.4f'))


# # # 【6】 GBDT(Gradient Boosting Decision Tree) Classifier
# # # 梯度增强决策树分类器
# from sklearn.ensemble import GradientBoostingClassifier
#
# GBDT = GradientBoostingClassifier()
# GBDT.fit(train_x, train_y)
# pred_test_GBDT = GBDT.predict(test_x)
# print('accuracyGBDT:',format(sm.accuracy_score(test_y, pred_test_GBDT),'.4f'))
# print('recall:',format(sm.recall_score(test_y, pred_test_GBDT,average='macro'),'.4f'))
# print('F1:',format(sm.f1_score(test_y, pred_test_GBDT, average='macro'),'.4f'))
#
# # 【7】 GaussianNB
# # # 高斯贝叶斯分类器
# from sklearn.naive_bayes import GaussianNB
#
# Gaussian = GaussianNB()
# Gaussian.fit(train_x, train_y)
# pred_test_Gaussian = Gaussian.predict(test_x)
# print('accuracyGaussian:',format(sm.accuracy_score(test_y, pred_test_Gaussian),'.4f'))
# print('recall:',format(sm.recall_score(test_y, pred_test_Gaussian,average='macro'),'.4f'))
# print('F1:',format(sm.f1_score(test_y, pred_test_Gaussian, average='macro'),'.4f'))
# #
# #
# # # 多项式贝叶斯分类器
# from sklearn.naive_bayes import MultinomialNB
#
# Multinomial = MultinomialNB()
# Multinomial.fit(train_x, train_y)
# pred_test_Multinomial = Multinomial.predict(test_x)
# print('MultinomialNB:',format(sm.accuracy_score(test_y, pred_test_Multinomial),'.4f'))
#
# # # 伯努利贝叶斯分类器
# from sklearn.naive_bayes import BernoulliNB
#
# Bernoulli = BernoulliNB()
# Bernoulli.fit(train_x, train_y)
# pred_test_BernoulliNB = Gaussian.predict(test_x)
# print('BernoulliNB:',format(sm.accuracy_score(test_y, pred_test_BernoulliNB),'.4f'))
#

### AdaBoost
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.datasets import make_gaussian_quantiles
#
# bdt = AdaBoostClassifier()
# bdt.fit(train_x, train_y)
# print(bdt.score(test_x, test_y))

### XGBClassifier
# from xgboost import XGBClassifier
# mo = XGBClassifier()
# mo.fit(train_x, train_y)
# pred_test_tre = mo.predict(test_x)
# print('tre:',format(mo.accuracy_score(test_y, pred_test_tre),'.4f'))