# coding: utf-8
import matplotlib.pyplot as plt
import os
import numpy as np
def readFlexData(filename):
    '''
    ;function: 读取五个传感器电压数据
    ;parameters: 
        filename: 存储五个传感器电压数据文件
    '''
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
    plt.savefig("../../data/validationFile/{}.png".format(savename))

def plotCompare(x1,y1,x2,y2,savename):
    '''
    ;function: 绘制多项式拟合效果图
    ;parameters: 
        flexData: 某一个传感器电压数据
        savename: 存储图片对于的文件名 “../../data/validationFile/{}.png".format(savename)
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
    plt.savefig("../../data/validationFile/{}.png".format(savename))


def saveBatchpic(folder):
    '''
    ;function: 对于不同状态下的传感器数据批处理作图 
    '''
    for folder1 in os.listdir(folder):
        if folder1[0]=='t':
            continue
        i=0
        for filename in os.listdir(os.path.join(folder,folder1)):
            data=readFlexData(os.path.join(folder,folder1,filename))
            plotLines(data,folder1+str(i))
            i=i+1





def fitSingleFlexData(onerecord):
    '''
    ;function: 输入某一个传感器校验数据，返回从0-180度对应的电压数据
    ;paremeters:
        onerecord: 某一个传感器带校验数据
        minA: 弯曲180度对应电压值
        maxB： 弯曲0度对应电压值
    '''
    onerecord.sort(reverse=False)
    beginIndex=0
    lastIndex=len(onerecord)-1
    minA=np.min(onerecord)
    maxB=np.max(onerecord)
    # for i in range(0,len(onerecord)-1):
    #     if onerecord[i]>minA and beginIndex==0:
    #         beginIndex=i
    #     if onerecord[i]<maxB and onerecord[i+1]>maxB:
    #         lastIndex=i+1
    # print(beginIndex,lastIndex,len(onerecord))
    # # plotLine(onerecord,str(413))        
    # onerecord=onerecord[beginIndex:lastIndex]
    return onerecord,minA

def getXYData(voltageData):
    '''
    ;function: 构造代拟合数据的y值，为与第一个值的插值；
    ;parameters: 
        voltageData: 从小到大 电压值序列， 其中对于弯曲度数  180--》0度
    ;return:
        
    '''
    yvalue = []
    for i in range(len(voltageData)):
        yvalue.append(voltageData[i]-voltageData[0] )
        # mid.append((1023 / value[i]- 1) * 2 - (1023 / value[0] - 1) * 2)
    xvalue=[]
    dist=180/len(yvalue)
    for i in range(0,len(yvalue)):
        xvalue.append(180-i*dist)
    #plotLine(yvalue,str(416))     
    return np.asarray(xvalue),np.asarray(yvalue)

# 使用非线性最小二乘法拟合
# 用指数形式来拟合
def func(x,a,b,c):
    # return np.log(b*x+c)+d
    return a*x*x+b*x+c
    #return a/(b+x)+c

from scipy.optimize import curve_fit
def draw_curve_fit(onerecord,discription="validationCompare"):
    '''
    ;function: 输入一组电压变化数据，进行二项式拟合处理
    ;parameters:
        onerecord: 某一个传感器通过  伸直 和 弯曲180度状态截取后的，升序排列的数据
    ;return: 
        a,b,c: 二项式拟合的三个系数
    '''
    x,y=getXYData(onerecord)
    popt, pcov = curve_fit(func, x, y)
    # popt里面是拟合系数，读者可以自己help其用法
    a = popt[0]
    b = popt[1]
    c = popt[2]
    yvals = func(x, a, b, c)
    plotCompare(x,y,x,yvals,"{}.png".format(discription))
    return a,b,c


def plotAllValidationFiles():
    folder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\validation"
    parameters=[]
    for file in os.listdir(folder):
        filename=os.path.join(folder,file)
        flexData=readFlexData(filename)
        plotLines(flexData,file)
        #绘制显示
        #使用  length*5*3  记录二项式系数
        for data in flexData:


def fitFlexDataHandle(filename):
    data=readFlexData(filename)
    if len(data)!=len(voltage0) or len(data)!=len(voltage180):
        print("error 数据长度不一样")
        return
    parameters=[]
    minList=[]
    for i in range(0,len(data)):
        onerecord,minA=fitSingleFlexData(data[i])
        parameter=draw_curve_fit(onerecord)
        parameters.append(parameter)
        minList.append(minA)
    return parameters,minList


from pynverse import inversefunc
def toangle_curve(flexdata,angle_parameter,minList):
    '''
    ;function: 利用之前拟合的曲线，将电压值转化为对于的 弯曲度
    ;parameters:
        flexdata: 五个弯曲传感器数据，格式类似[461.3366336633662, 503.73125884016974, 457.57142857142856, 455.8712871287128, 439.36067892503536]
        angle_parameter: 五个弯曲传感器对于的 二项式曲线拟合系数 a,b,c 有五组系数构成列表
        minList: 五个弯曲传感器弯曲180度对于的电压值，用来做区间映射处理
    '''
    angle = []
    for i in range(len(flexdata)):
        cube = (lambda x: angle_parameter[i][0]*x*x +  angle_parameter[i][1]*x + angle_parameter[i][2])
        # cube = (lambda x: np.log(angle_parameter[i][0] * x + angle_parameter[i][1]) + angle_parameter[i][2])
        invcube = inversefunc(cube, y_values=flexdata[i]-minList[i])
        for i in [invcube.tolist()]:   #保证角度在 0-180度区间内
            if i>180:
                i=180
            if i<0:
                i=0
            angle.append(i)
    return angle

def validationFunctionTest():
    '''
    ;function: validation function 测试，测试没有问题 
    '''
    parameters,minList=fitFlexDataHandle("../../data/validationFile/validation{}.txt".format(str(4)),[251.77369165487974, 326.2489391796322, 276.42149929278645, 318.42715700141446, 274.3097595473833],[461.3366336633662, 503.73125884016974, 457.57142857142856, 455.8712871287128, 439.36067892503536])
    print("parameters",parameters)
    print("minList",minList)
    print("angle",toangle_curve([251.77369165487974, 326.2489391796322, 276.42149929278645, 318.42715700141446, 274.3097595473833],parameters,minList))

#validationFunctionTest()
#saveBatchpic("../../data/flexSensor/")
#data=readFlexData(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\flexSensor\blend\1617933954.8908825origin.txt")
#plotLines(data)
#data=readFlexData(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\flexSensor\static90\1617934261.1825488origin.txt")
#data=readFlexData("../../data/validationFile/validation{}.txt".format(str(4)))

#plotLines(data,str(41))
# data[3],346.2616690240453,455.5516265912306
# data[3],318.42715700141446,455.8712871287128
#fitSingleFlexData(data[2],276.42149929278645,457.57142857142856)
# tempdata=fitSingleFlexData(data[2],276.42149929278645,457.57142857142856)   #RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800.
# a,b,c=draw_curve_fit(tempdata)
# cube = (lambda x: a*x*x +  b*x + c)
#         # cube = (lambda x: np.log(angle_parameter[i][0] * x + angle_parameter[i][1]) + angle_parameter[i][2])
# invcube = inversefunc(cube, y_values=[252.85714285714286-276.42149929278645])
# print(invcube.tolist())

def avgFilter(oneData):
    '''
    ;function: 滑动平均效果绘图
    ;paremeters:
        oneData: 某一个传感器带校验数据
    '''
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

def filterComparision(data):
    '''
    ;function: 不同种过滤算法对比
    ;paremeters:
        data: 传感器带校验数据
    '''
    avgFilter(data[1])
    kalmanFilter(data[1])
    UKFhandle(data[1])
    filter_low(data[1])

