# coding: utf-8
import matplotlib.pyplot as plt
import os

def readFlexData(filename):
    dicFlex=[]
    with open(filename,'r') as fileOp:
        lines=fileOp.readlines()
        if(len(lines)!=5):
            return Null
        else:
            for line in lines:
                dicFlex.append([float(i) for i in line.split(',')])
        fileOp.close()
    return dicFlex

def plotLines(flexData,savename):
    figsize = 9, 9
    figure, ax = plt.subplots(figsize=figsize)
    # 在同一幅图片上画两条折线
    A, = plt.plot(flexData[0], '-r', label='A', linewidth=5.0)
    B, = plt.plot(flexData[1], 'b-.', label='B', linewidth=5.0)
    C, = plt.plot(flexData[2], '-k.', label='C', linewidth=5.0)
    D, = plt.plot(flexData[3], 'm-.', label='D', linewidth=5.0)
    E, = plt.plot(flexData[4], 'g-.', label='E', linewidth=5.0)
    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 23,
            }
    legend = plt.legend(handles=[A, B,C,D,E], prop=font1)
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=23)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print labels
    [label.set_fontname('Times New Roman') for label in labels]
    # 设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 30,
            }
    plt.xlabel('Timestamps (ms)', font2)
    plt.ylabel('Voltage (V)', font2)
    plt.savefig("../../data/flexSensor/temppic/{}.png".format(savename))


def saveBatchpic(folder):
    for folder1 in os.listdir(folder):
        if folder1[0]=='t':
            continue
        i=0
        for filename in os.listdir(os.path.join(folder,folder1)):
            data=readFlexData(os.path.join(folder,folder1,filename))
            plotLines(data,folder1+str(i))
            i=i+1

#saveBatchpic("../../data/flexSensor/")
#data=readFlexData(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\flexSensor\blend\1617933954.8908825origin.txt")
#plotLines(data)
data=readFlexData(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\flexSensor\static90\1617934261.1825488origin.txt")

from filterOp import *
def avgFilter(oneData):
    afterFilter=[]
    movAvg=MovAvg(15)
    for i in oneData:
        afterFilter.append(movAvg.update(i))
    figsize = 9, 9
    figure, ax = plt.subplots(figsize=figsize)
    # 在同一幅图片上画两条折线
    A, = plt.plot(oneData[15:], '-r', label='A', linewidth=5.0)
    B, = plt.plot(afterFilter[15:], 'b-.', label='B', linewidth=5.0)
    font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 23,
            }
    legend = plt.legend(handles=[A, B], prop=font1)
    plt.show()

def filterComparision():
    avgFilter(data[1])
    kalmanFilter(data[1])
    UKFhandle(data[1])
    filter_low(data[1])

