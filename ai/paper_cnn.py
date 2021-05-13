#-*- coding:utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.utils import plot_model
from data_loadKera import SequenceData
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import os

#{'y': 0, 'i': 1, 'r': 2, 's': 3, 'm': 4, 'next': 5, 'q': 6, 'P': 7, 'o': 8, 'v': 9, 'down': 10, 'square': 11, 'previous': 12, 'z': 13, 'rectangle': 14, 'a': 15, 'k': 16, 'c': 17, 'j': 18, 'u': 19, 'x': 20, 'b': 21, 'd': 22, 'l': 23, 'f': 24, 'g': 25, 't': 26, 'star': 27, 'n': 28, 'h': 29, 'w': 30, 'e': 31}
# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        print("train acc",self.accuracy[loss_type])
        print("train loss",self.losses[loss_type])
        print("val acc",self.val_acc[loss_type])
        print("val loss",self.val_loss[loss_type])
        np.savetxt("../data/train_acc.txt", self.accuracy[loss_type], fmt="%s", delimiter=' ')
        np.savetxt("../data/train_loss.txt", self.losses[loss_type], fmt="%s", delimiter=' ')
        np.savetxt("../data/val_acc.txt", self.val_acc[loss_type], fmt="%s", delimiter=' ')
        np.savetxt("../data/val_loss.txt", self.val_loss[loss_type], fmt="%s", delimiter=' ')

        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
         # val_acc
        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        # val_loss
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig("result.png",dpi=100)

def baseline_model():
    #create model
    model = Sequential()                                              #(2781, 96, 96, 1)  (2781, 96, 96)
    model.add(Convolution2D(50, 15, 15, border_mode='valid', input_shape=(96, 96, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(25, 10, 10, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(29,activation='softmax'))              # 26 分类
    #Compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
if __name__=='__main__':
    #freeze_support()

    #读取数据 data/kinect/test_3/
    #basePath = '../data/final/train/'
    #validPath='../data/final/valid/'

    basePath = '../data/CharData/train/'
    validPath='../data/CharData/valid/'

    numclass=29 #分类输出个数
    batch_size = 50
    valid_batchsize=50
    sequence_data = SequenceData(basePath, batch_size,numclass)
    valid_data=SequenceData(validPath,valid_batchsize,numclass)
    steps =sequence_data.__len__()
    validsteps=valid_data.__len__()


    #模型回调函数
    his = LossHistory()
    filepath="./summodel1/model_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=False,
    mode='max')
    callbacks_list = [checkpoint,his]
    #定义模型
    model = baseline_model()
    #model.load_weights("model_04-1.00.hdf5")
    loss = model.fit_generator(sequence_data, steps_per_epoch=steps,validation_data=valid_data,validation_steps= validsteps, epochs=5, verbose=1,shuffle=True,callbacks=callbacks_list)
    modelPath = 'cnn_model1.h5'
    model.summary()
    #保存模型
    model.save(modelPath)
    his.loss_plot('epoch')
    his.loss_plot('batch')
