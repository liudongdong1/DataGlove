import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          size=None):
    """ (Copied from sklearn website)
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Display normalized confusion matrix ...")
    else:
        print('Display confusion matrix without normalization ...')

    # print(cm)

    fig, ax = plt.subplots()
    if size is None:
        size = (12, 8)
    fig.set_size_inches(size[0], size[1])

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim([-0.5, len(classes)-0.5])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm


# Drawings ==============================================================

def draw_action_result(img_display, id, skeleton, str_action_label):
    font = cv2.FONT_HERSHEY_SIMPLEX

    minx = 999
    miny = 999
    maxx = -999
    maxy = -999
    i = 0
    NaN = 0

    while i < len(skeleton):
        if not(skeleton[i] == NaN or skeleton[i+1] == NaN):
            minx = min(minx, skeleton[i])
            maxx = max(maxx, skeleton[i])
            miny = min(miny, skeleton[i+1])
            maxy = max(maxy, skeleton[i+1])
        i += 2

    minx = int(minx * img_display.shape[1])
    miny = int(miny * img_display.shape[0])
    maxx = int(maxx * img_display.shape[1])
    maxy = int(maxy * img_display.shape[0])

    # Draw bounding box
    # drawBoxToImage(img_display, [minx, miny], [maxx, maxy])
    img_display = cv2.rectangle(
        img_display, (minx, miny), (maxx, maxy), (0, 255, 0), 4)

    # Draw text at left corner
    box_scale = max(
        0.5, min(2.0, (1.0*(maxx - minx)/img_display.shape[1] / (0.3))**(0.5)))
    fontsize = 1.4 * box_scale
    linewidth = int(math.ceil(3 * box_scale))

    TEST_COL = int(minx + 5 * box_scale)
    TEST_ROW = int(miny - 10 * box_scale)
    # TEST_ROW = int( miny)
    # TEST_ROW = int( skeleton[3] * img_display.shape[0])

    img_display = cv2.putText(
        img_display, "P"+str(id % 10)+": "+str_action_label, (TEST_COL, TEST_ROW), font, fontsize, (0, 0, 255), linewidth, cv2.LINE_AA)


def add_white_region_to_left_of_image(img_disp):
    r, c, d = img_disp.shape
    blank = 255 + np.zeros((r, int(c/4), d), np.uint8)
    img_disp = np.hstack((blank, img_disp))
    return img_disp



def plotLines(flexData,savename):
    '''
    ;function: 绘制五个传感器电压数据曲线图，于一张图片上
    ;parameters: 
        flexData: 五个个传感器电压数据列表
        savename: 存储图片对于的文件名"../../data/validationFile/{}.png".format(savename)
    '''
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
    #plt.savefig("../../data/flexSensor/temppic/{}.png".format(savename))
    plt.savefig("./pngresult/validation/{}.png".format(savename))





def plotLine(flexData,savename):
    '''
    ;function: 绘制单个传感器电压数据曲线图
    ;parameters: 
        flexData: 某一个传感器电压数据
        savename: 存储图片对于的文件名 “../../data/validationFile/{}.png".format(savename)
    '''
    figsize = 9, 9
    figure, ax = plt.subplots(figsize=figsize)
    # 在同一幅图片上画两条折线
    A, = plt.plot(flexData, '-r', label='A', linewidth=5.0)
    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 23,
            }
    legend = plt.legend(handles=[A], prop=font1)
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
    #plt.savefig("../../data/flexSensor/temppic/{}.png".format(savename))
    plt.savefig("./pngresult/validation/{}.png".format(savename))

def plotCompare(x1,y1,x2,y2,savename):
    '''
    ;function: 绘制多项式拟合效果图
    ;parameters: 
        flexData: 某一个传感器电压数据
        savename: 存储图片对于的文件名 “./pngresult/validation/{}.png".format(savename)
    '''
    figsize = 9, 9
    figure, ax = plt.subplots(figsize=figsize)
    # 在同一幅图片上画两条折线
    A, = plt.plot(x1,y1, '-r', label='origin', linewidth=5.0)
    B, = plt.plot(x2,y2, 'b-.', label='fitcurve', linewidth=5.0)
    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 23,
            }
    legend = plt.legend(handles=[A,B], prop=font1)
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
    plt.xlabel('degree', font2)
    plt.ylabel('Voltage (V)', font2)
    #plt.savefig("../../data/flexSensor/temppic/{}.png".format(savename))
    plt.savefig("./pngresult/validation/{}.png".format(savename))