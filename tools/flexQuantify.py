# coding: utf-8
import matplotlib.pyplot as plt
import os
import numpy as np
from lib_txtIO import *
from lib_plot import *


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


def getXYData(voltageData):
    '''
    ;function: 构造代拟合数据的y值，为与第一个值的插值；
    ;parameters: 
        voltageData: 从小到大 电压值序列， 其中对于弯曲度数  180--》0度
    ;return:
        x: np   弯曲度数  180-0度
        y: np   对应弯曲电压
        z: float 最小电压，对应最大弯曲度数180
    '''
    voltageData.sort(reverse=False)
    yvalue=[]
    for i in range(len(voltageData)):
        temp=abs(voltageData[i]-voltageData[0])
        yvalue.append(temp)
    max=np.max(yvalue)
    tvalue=[]
    for i in range(0,len(yvalue)):
        tvalue.append(yvalue[i])
        if max - yvalue[i]<10:
            break
    tvalue.append(yvalue[-1])
    dist=180/len(tvalue)
    xvalue=[]
    for i in range(0,len(tvalue)):
        xvalue.append(180-i*dist)
    #plotLine(yvalue,str(416))     
    return np.asarray(xvalue),np.asarray(tvalue),voltageData[0]

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
        a,b,c: 二项式拟合的三个系数   a*x*x+b*x+c
        minvalue: 最小电压，用于基准做差
    '''
    x,y,minvalue=getXYData(onerecord)
    popt, pcov = curve_fit(func, x, y)
    # popt里面是拟合系数，读者可以自己help其用法
    a = popt[0]
    b = popt[1]
    c = popt[2]

    # yvals = func(x, a, b, c)
    # plotCompare(x,y,x,yvals,"{}.png".format(discription))

    return a,b,c,minvalue




def fitFlexDataHandle(filename):
    '''
        function: 进行矫正，获取二项式系数a*x*x+b*x+c： a, b, c, 最小电压
        input: 
            filename: 存储校准数据txt文件目录
        return:
            parameters: 5*4, 分别五个手指， a,b,c,reference(180对应的最小电压)
    '''
    data=readFlexData(filename)
    if len(data)!=5 :
        print("error 数据长度不一样")
        return
    parameters=[]
    for i in range(0,len(data)):
        parameter=draw_curve_fit(data[i],filename[-19:-9])
        parameters.append(parameter)
    return parameters


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

def drawSingleValidationbefore(filename):
    folder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\validation"
    filename=os.path.join(folder,filename)
    flexData=readFlexData(filename)
    for data in flexData:
        x,y=getXYData(data)
        plotLine(y,"temp")

def drawSingleValidation(filename):
    folder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\validation"
    filename=os.path.join(folder,filename)
    param=fitFlexDataHandle(filename)
    print(param)

#drawSingleValidation("validation1621863387.0423715.txt")
def plotAllValidationFiles():
    folder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\validation"
    parameters=[]
    for file in os.listdir(folder):
        filename=os.path.join(folder,file)
        param=fitFlexDataHandle(filename)
        parameters.append(param)
    #Save_list(parameters,"./pngresult/validationparam.txt")
    data=np.asarray(parameters)
    print(data.shape)

    for i in range(0,5):
        for j in range(0,4):
            temp=[]
            for k in range(0,52):
                temp.append(data[k][i][j])
            plotLine(temp,"parame_{}_flex{}.png".format(j,i))



#plotAllValidationFiles()
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

