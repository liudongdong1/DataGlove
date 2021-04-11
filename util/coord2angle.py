
import numpy as np
from ast import literal_eval
import os
# 算手指弯曲度
# 配对表：1,2-3,4   5,6-7,8   9,10 - 11,12     13,14 - 15,16     17,18 - 19,20

def calculateAngle(a,b):
        '''
            求a 和 b列表的余弦值 
        '''
        a_l = sum([x*x for x in a])
        b_l = sum([x*x for x in b])
        if a_l * b_l == 0:
            return 0
        uper = 0
        for i in range(len(a)):
            uper = uper + a[i]*b[i]
        angle_cos = uper / np.sqrt(a_l * b_l)
        # print(angle_cos)
        angle = np.arccos(angle_cos)
        return angle

def meanfinger(finger_write):
    '''
        同一手势不同图像特征向量取平均，每张图像对应特征向量与平均值作差，将差值超过阈值的数据剔除。本文设立阈值为 。 
    '''
    a = []
    b = []
    for x in finger_write:
        a.append(x[:-2])
    mid = np.average(a, 0)   #跨列进行计算
    for k in range(len(a)):
        for i in range(len(a[k])):
            if np.abs(a[k][i]-mid[i])>=np.pi/6:
                b.append(k)
                break
    for i in range(len(b)):
        finger_write.pop(b[len(b)-i-1])
    return finger_write

def toangle5(filename_all, filename_last):
    '''
    ;function: 将mediapipe 检测出来的3D坐标进行向量化处理
    ；parameters:
        filename_all: 3d坐标，hand，label
        filename_last: 存储最终角度文件 
    '''
    with open(filename_all, 'r') as fl2:
        f2 = fl2.readlines()
        final_lrdata = []
        # 将转换后的列表从str转换回真正的list '[]' - []
        for x in f2:
            final_lrdata =literal_eval(x)
    # print(len(final_lrdata[0]))
    # print(len(final_lrdata[0][0]))


    finger_write = []
    for k in range(len(final_lrdata)):
        finger_list = []
        # 2 为 步长
        for i in range(1, 21, 2):
            temp = [final_lrdata[k][i][j] - final_lrdata[k][i + 1][j] for j in range(len(final_lrdata[k][0]))]
            finger_list.append(temp)

        finger_last = []

        for i in range(0, len(finger_list), 2):
            temp = np.pi - calculateAngle(finger_list[i], finger_list[i + 1])  # pi 减去计算的余弦值
            temp = float(format(temp,'.4f'))
            finger_last.append(temp)

        finger_last.append(final_lrdata[k][-2])   #left or right hand
        finger_last.append(final_lrdata[k][-1])   #label
        finger_write.append(finger_last)          #一个label的总数据

    finger_m = meanfinger(finger_write)
    # finger_m = finger_write


    with open(filename_last, 'w') as fl3:
        fl3.write(str(finger_m))

def transforAll3DData(basefolder):
        '''
            functions: 利用media匹配处理成文本数据里，批处理提取格式化数据，坐标，左右手
            parameters: 
                basefolder: default r"..\..\data\temp\picFlex\", 其中该目录下面有有A-zlabel 文件，每一个label文件下面是几个txt文件：
                origindatafile: mediapipe提取的 手部坐标点数据
                originleftright: mediapipe 提取的左右手的状态
                coordinate ： 最终格式化 [x,y,z], 手部状态数据
                listtestlr_f ： 最终手部状态数据
                allcoordinate ： 最终格式化 [x,y,z], 手部状态数据
                angle: 一个label对应的所有 角度数据

        '''
        labels=os.listdir(basefolder)
        print("所有的label为： ",labels)
        for label in labels:
            print("handle coordinate-angle folder :", label)
            file4=os.path.join(basefolder,label,"angle")
            file5=os.path.join(basefolder,label,"allcoordinate")
            toangle5(file5,file4)
folder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\word"
transforAll3DData(folder)


# a = [1,3,2,6,'Right','A']
# b1 = [1,1,1,8,'Left','A']
# b2 = [1,1,1,8,'Right','B']
# b3 = [1,1,1,8,'Right','A']
# b4 = [1,1,1,8,'Right','A']
# # c = [a[i] - b[i] for i in range(len(a))]
# # print(-a)
# d = [a,b1,b2,b3,b4]
# print('origin',d)
# print(meanfinger(d))
# filename_last=r"C:\project\ASL\ArduinoProject\VR-Glove\DashBoard\data\temp\picFlex\A\angle"
# with open(filename_last, 'r') as fl2:
#     f2 = fl2.readlines()

#     final_lrdata = []
#     # 将转换后的列表从str转换回真正的list '[]' - []
#     for x in f2:
#         final_lrdata=literal_eval(x)
#     print(len(final_lrdata))