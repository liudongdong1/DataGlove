# coding: utf-8
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.lib_txtIO import *
from tools.lib_plot import *
import pickle



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
    # np.savetxt("x.txt", x,fmt='%d',delimiter=',')
    # np.savetxt("xy.txt", y,fmt='%d',delimiter=',')
    # np.savetxt("xy1.txt", yvals,fmt='%d',delimiter=',')
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
def toangle_curve(flexdata,angle_parameter):
    '''
    ;function: 利用之前拟合的曲线，将电压值转化为对于的 弯曲度
    ;parameters:
        flexdata: 五个弯曲传感器数据，格式类似[461.3366336633662, 503.73125884016974, 457.57142857142856, 455.8712871287128, 439.36067892503536]
        angle_parameter: 五个弯曲传感器对于的 二项式曲线拟合系数 a,b,c 有五组系数构成列表
        minList: 五个弯曲传感器弯曲180度对于的电压值，用来做区间映射处理
    '''
    #print("flexdata:",flexdata,"\nangle_parameter",angle_parameter)

    
    angle = []
    for i in range(len(flexdata)):
        cube = (lambda x: angle_parameter[i][0]*x*x +  angle_parameter[i][1]*x + angle_parameter[i][2])
        #invcube = inversefunc(cube, y_values=[t-angle_parameter[i][3] for t in flexdata[i]])
        #print(flexdata[i]-angle_parameter[i][3])
        invcube = inversefunc(cube, y_values=[flexdata[i]-angle_parameter[i][3]])
        #print(type(invcube),invcube.shape)
        for i in range(0,len(invcube)):   #保证角度在 0-180度区间内
            if invcube[i]>180:
                invcube[i]=180
            if invcube[i]<0:
                invcube[i]=0
        angle.append(invcube[0])
    #print(angle)
    return angle

import matplotlib.pyplot as plt
def dataDescription():
    basefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\charGen\train"
    fig,axes=plt.subplots(nrows=2,ncols=13,figsize=(130,30))
    print(type(axes))
    i=0
    blotplist=[]
    for tempfile in os.listdir(basefolder):
        filename=os.path.join(basefolder,tempfile)
        data=numpyload(filename).T
        print("label_{}.shape{}. {}".format(tempfile,data.shape[0],data.shape[1]))
        #print(data.tolist())
        bplot1=axes[int(i/13),i%13].boxplot(data.tolist(),
                       vert=True,
                       patch_artist=True)
        axes[int(i/13),i%13].yaxis.grid(True) #在y轴上添加网格线
        ##axes[int(i/6),(i-int(i/6))%4].set_xticks(["A","B","C","D","E"] ) #指定x轴的轴刻度个数
        axes[int(i/13),i%13].set_xlabel('xlabel') #设置x轴名称
        axes[int(i/13),i%13].set_ylabel('ylabel') #设置y轴名称
        blotplist.append(bplot1)
        i=i+1
    colors = ['pink', 'lightblue', 'lightgreen','red','orange']
    print("blot length:",len(blotplist))
    for bplot in blotplist:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.savefig("charDescription.png")



def digitDataDescription():
    basefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\digitGen\train"
    fig,axes=plt.subplots(nrows=2,ncols=5,figsize=(100,30))
    print(type(axes))
    i=0
    blotplist=[]
    for tempfile in os.listdir(basefolder):
        filename=os.path.join(basefolder,tempfile)
        data=numpyload(filename).T
        print("label_{}.shape{}. {}".format(tempfile,data.shape[0],data.shape[1]))
        #print(data.tolist())
        bplot1=axes[int(i/5),i%5].boxplot(data.tolist(),
                       vert=True,
                       patch_artist=True)
        axes[int(i/5),i%5].yaxis.grid(True) #在y轴上添加网格线
        ##axes[int(i/6),(i-int(i/6))%4].set_xticks(["A","B","C","D","E"] ) #指定x轴的轴刻度个数
        axes[int(i/5),i%5].set_xlabel(tempfile) #设置x轴名称
        axes[int(i/5),i%5].set_ylabel('ylabel') #设置y轴名称
        blotplist.append(bplot1)
        i=i+1
    colors = ['pink', 'lightblue', 'lightgreen','red','orange']
    print("blot length:",len(blotplist))
    for bplot in blotplist:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.savefig("CharPictureTrain.png")



def digitPictureDescription():
    basefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\digit"
    fig,axes=plt.subplots(nrows=2,ncols=5,figsize=(100,30))
    print(type(axes))
    i=0
    blotplist=[]
    print(os.listdir(basefolder))
    for tempfile in os.listdir(basefolder):
        filename=os.path.join(basefolder,tempfile)
        data=readData(filename)
        data=np.array(data).T
        print("label_{}.shape{}".format(tempfile,data.shape))
        #print(data.tolist())
        bplot1=axes[int(i/5),i%5].boxplot(data.tolist(),
                       vert=True,
                       patch_artist=True)
        axes[int(i/5),i%5].yaxis.grid(True) #在y轴上添加网格线
        ##axes[int(i/6),(i-int(i/6))%4].set_xticks(["A","B","C","D","E"] ) #指定x轴的轴刻度个数
        axes[int(i/5),i%5].set_xlabel(tempfile) #设置x轴名称
        axes[int(i/5),i%5].set_ylabel('ylabel') #设置y轴名称
        blotplist.append(bplot1)
        i=i+1
    colors = ['pink', 'lightblue', 'lightgreen','red','orange']
    print("blot length:",len(blotplist))
    for bplot in blotplist:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.savefig("digitPictureTrain.png")

#-----------------函数功能测试部分代码---------
def drawSingleValidationbefore(filename):
    folder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\validation"
    for file in os.listdir(folder):
        fullfilename=os.path.join(folder,file)
        flexData=readFlexData(fullfilename)
        flexData=np.asarray(flexData)
        print(flexData.shape)
        plotLines(flexData,file.split(".")[0])

distr_digit={"one":[100,140,110,150,100,150,0,20,30,70],"two":[70,110,90,130,0,30,0,20,30,70],"three":[0,20,0,20,0,20,90,110,20,50],"four":[0,20,0,20,0,20,0,20,40,85],'five':[0,20,0,20,0,20,0,20,0,20],"six":[0,20,70,90,100,120,100,120,0,20],"seven":[100,130,80,110,30,50,10,50,0,20],"eight":[100,130,80,130,80,140,0,20,0,20],"nine":[80,100,90,110,100,130,60,80,50,70]}
#distr_char={"A":[110,130,100,125,100,130,90,125,0,30],"B":[0,20,0,30,0,30,0,30,60,85],"C":[50,85,60,80,70,90,40,60,30,45],"D":[110,130,110,130,110,140,0,20,30,60],"E":[90,120,90,110,100,115,105,140,70,90],"F":[0,20,0,20,0,20,60,80,20,45],"G":[100,140,90,130,120,145,0,20,40,70],"H":[90,120,100,120,0,30,0,20,40,70],"I":[0,20,70,90,100,130,100,140,50,70],"J":[0,20,70,90,100,120,0,20,0,20],"K":[80,120,80,110,0,20,0,20,0,20],"L":[100,130,100,130,110,140,0,20,0,20],"M":[90,110,70,80,80,95,100,120,70,100],"N":[80,95,45,85,65,95,75,100,30,60],"O":[50,80,60,80,70,100,60,90,40,60],"P":[70,110,70,120,30,60,0,20,0,20],"Q":[70,130,70,130,80,130,0,30,0,20],"R":[60,100,80,100,0,20,0,20,30,50],"S":[90,120,100,120,110,140,110,140,60,90],"T":[40,90,60,90,70,110,55,85,30,45],"U":[60,100,90,110,0,20,0,20,40,70],"V":[50,100,90,110,0,20,0,20,0,20,40,70],"W":[90,120,0,30,0,20,0,20,40,60],"X":[60,90,110,130,10,140,60,90,30,50],"Y":[0,20,60,80,90,110,80,110,0,20],"Z":[0,20,0,20,0,20,0,20,0,20]}
distr_char={"A":[110,130,100,125,100,130,90,125,0,30],"B":[0,20,0,30,0,30,0,30,60,85],"C":[50,85,60,80,70,90,40,60,30,45],"D":[110,130,110,130,110,140,0,20,30,60],"E":[90,120,90,110,100,115,105,140,70,90],"F":[0,20,0,20,0,20,60,80,20,45],"G":[100,140,90,130,120,145,0,20,40,70],"H":[90,120,100,120,0,30,0,20,40,70],"I":[0,20,70,90,100,130,100,140,50,70],"J":[0,20,70,90,100,120,0,20,0,20],"K":[80,120,80,110,0,20,0,20,0,20],"L":[100,130,100,130,110,140,0,20,0,20],"M":[90,110,70,80,80,95,100,120,70,100],"N":[80,95,45,85,65,95,75,100,30,60],"O":[50,80,60,80,70,100,60,90,40,60],"P":[70,110,70,120,30,60,0,20,0,20],"Q":[70,130,70,130,80,130,0,30,0,20],"R":[60,100,80,100,0,20,0,20,30,50],"S":[90,120,100,120,110,140,110,140,60,90],"T":[40,90,60,90,70,110,55,85,30,45],"U":[0,20,0,20,0,20,0,20,40,85],"V":[50,100,90,110,0,20,0,20,0,20,40,70],"W":[90,120,0,30,0,20,0,20,40,60],"X":[60,90,110,130,10,140,60,90,30,50],"Y":[0,20,60,80,90,110,80,110,0,20],"Z":[0,20,0,20,0,20,0,20,0,20]}

def digitGen(count=100):
    basefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\genData2\digitTrain"
    basefolder1=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\genData2\digit1"
    for key,value in distr_digit.items():
        data=np.loadtxt(os.path.join(basefolder,"{}.txt".format(key)),delimiter=',')
        print("handle file{}".format(os.path.join(basefolder,"{}.txt".format(key))))
        for i in range(count):
            data=np.append(data,[[np.random.randint(value[0],value[1]),np.random.randint(value[2],value[3]),np.random.randint(value[4],value[5]),np.random.randint(value[6],value[7]),np.random.randint(value[8],value[9])]],axis=0)
        np.savetxt(os.path.join(basefolder1,"{}.txt".format(key)),data,delimiter=',',fmt='%d')

def charGen(count=100):
    basefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\genData2\charTrain"
    basefolder1=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\genData2\char1"
    for key,value in distr_char.items():
        data=np.loadtxt(os.path.join(basefolder,"{}.txt".format(key)),delimiter=',')
        print("handle file{}".format(os.path.join(basefolder,"{}.txt".format(key))))
        for i in range(count):
            data=np.append(data,[[np.random.randint(value[0],value[1]),np.random.randint(value[2],value[3]),np.random.randint(value[4],value[5]),np.random.randint(value[6],value[7]),np.random.randint(value[8],value[9])]],axis=0)
        np.savetxt(os.path.join(basefolder1,"{}.txt".format(key)),data,delimiter=',',fmt='%d')

def DataGenarate():
    basefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\genData"
    param=fitFlexDataHandle(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\validation\20.txt")
    print("self.parameters",param)
    savefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\genData1"
    for folder in os.listdir(basefolder):
        for tfolder in os.listdir(os.path.join(basefolder,folder)):
            tfullfolder=os.path.join(basefolder,folder,tfolder)
            tempdata=[]
            for file in os.listdir(tfullfolder):
                data=readFlexData(os.path.join(tfullfolder,file))
                data=np.asarray(data).T   # 181*5
                #for i in range(0,5):
                #    data[:,i]=data[:,i]*data[:,i]*param[i][0]+param[i][1]*data[:,i]+param[i][2]+param[i][3]
                if len(tempdata)==0:
                    tempdata=data
                else:
                    tempdata=np.row_stack((tempdata,data))
            #     tempdata.append(data)
            # tempdata=np.asarray(tempdata)
            print(tempdata.shape)
            np.savetxt(os.path.join(savefolder,folder,"{}.txt".format(tfolder)),tempdata,delimiter=',',fmt='%d')
    # validfolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\validation"
    # for folder in os.listdir(savefolder):
    #     print(folder)
    #     for file in os.listdir(os.path.join(savefolder,folder)):
    #         data=np.loadtxt(os.path.join(savefolder,folder,file),delimiter=',')
    #         #print("data:",data.shape)
    #         tempdata=[]
    #         for tfile in os.listdir(validfolder):
    #             param=fitFlexDataHandle(os.path.join(validfolder,tfile))
    #             for i in range(len(data)):
    #                 data[i,:]=toangle_curve(data[i,:],param)
    #             #print(type(data))
    #             if len(tempdata)==0:
    #                 tempdata=data
    #             else:
    #                 tempdata=np.row_stack((tempdata,data))
    #         print(tempdata.shape)   
    #         np.savetxt(os.path.join(savefolder,folder,file),tempdata,delimiter=',',fmt='%d')
            

def drawSingleValidation(filename):
    folder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\validation"
    filename=os.path.join(folder,filename)
    param=fitFlexDataHandle(filename)
    print(param)


#drawSingleValidation("validation1621863387.0423715.txt")

def plotAllValidationFiles():
    # folder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\validation"
    # parameters=[]
    # for file in os.listdir(folder):
    #     filename=os.path.join(folder,file)
    #     param=fitFlexDataHandle(filename)
    #     parameters.append(param)
    # ###Load into file
    # with open("./pngresult/parameters.pkl","wb") as f:
    #     pickle.dump(parameters,f)
    with open("./pngresult/parameters.pkl","rb") as f:
        parameters = pickle.load(f)
    print(parameters)
    #handleSingleFile(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\char26Large\A","1621863998.7245626.txt",parameters)
    genALLFolder(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\digitLarge",parameters)
    #Save_list(parameters,"./pngresult/validationparam.txt")
    # data=np.asarray(parameters)
    # print(data.shape)

    # for i in range(0,5):
    #     for j in range(0,4):
    #         temp=[]
    #         for k in range(0,52):
    #             temp.append(data[k][i][j])
    #         plotLine(temp,"parame_{}_flex{}.png".format(j,i))




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

# 生成弯曲传感器数据
def handleSingleFile(folder, filename,parameters):
    filename=os.path.join(folder,filename)
    flexdata=readFlexData(filename)
    OneFileData=[]
    for parameter in parameters:
        data=np.vstack(toangle_curve(flexdata,parameter)).T
        #print(type(data))
        if len(OneFileData)==0:
            OneFileData=data
        else:
            OneFileData=np.row_stack((OneFileData,data))  # 按列拼接，添加在列尾部
    #print(type(OneFileData),OneFileData.shape)
    return OneFileData


def genAll(folder,parameters):
    OneFolderData=[]
    for file in os.listdir(folder):
        data=handleSingleFile(folder,file,parameters)
        if len(OneFolderData)==0:
           OneFolderData=data
        else:
            OneFolderData=np.row_stack((OneFolderData,data))  # 按列拼接，添加在列尾部
    return OneFolderData

def genALLFolder(base,parameters):
    savefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\digitGen"
    for folder in os.listdir(base):
        dataset=genAll(os.path.join(base,folder),parameters)
        np.random.shuffle(dataset)
        print("label{} shape{}".format(type(dataset),dataset.shape))
        length=dataset.shape[0]
        numpysave(dataset[:int(length*0.6)],os.path.join(savefolder,"train","{}.txt".format(folder)))
        numpysave(dataset[int(length*0.6):int(length*0.9)],os.path.join(savefolder,"valid","{}.txt".format(folder)))
        numpysave(dataset[int(length*0.9):],os.path.join(savefolder,"test","{}.txt".format(folder)))



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

#绘制数据分布箱型图
def dataPictureDescription():
    basefolder=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\char"
    fig,axes=plt.subplots(nrows=2,ncols=13,figsize=(130,30))
    print(type(axes))
    i=0
    blotplist=[]
    print(os.listdir(basefolder))
    for tempfile in os.listdir(basefolder):
        filename=os.path.join(basefolder,tempfile)
        data=readData(filename)
        data=np.array(data).T
        print("label_{}.shape{}".format(tempfile,data.shape))
        #print(data.tolist())
        bplot1=axes[int(i/13),i%13].boxplot(data.tolist(),
                       vert=True,
                       patch_artist=True)
        axes[int(i/13),i%13].yaxis.grid(True) #在y轴上添加网格线
        ##axes[int(i/6),(i-int(i/6))%4].set_xticks(["A","B","C","D","E"] ) #指定x轴的轴刻度个数
        axes[int(i/13),i%13].set_xlabel('xlabel') #设置x轴名称
        axes[int(i/13),i%13].set_ylabel('ylabel') #设置y轴名称
        blotplist.append(bplot1)
        i=i+1
    colors = ['pink', 'lightblue', 'lightgreen','red','orange']
    print("blot length:",len(blotplist))
    for bplot in blotplist:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.savefig("charPictureDescription.png")

#弯曲传感器转化测试
def bendTest():
    flexdata=[433.0, 456.0, 432.0, 404.0, 427.0] 
    angle_parameter=[(-0.0054964754572731514, -0.4592414766674026, 225.19958929017517, 195.6), (0.0012542545672676875, -1.446178097700333, 206.0764482034097, 258.1), (-0.003424300528683588, -0.5714580328295418, 185.99388661641942, 235.7), (-0.003333955041990531, -0.4951182914559451, 169.19466977840358, 226.9), (0.001063238985811843, -1.371116869711162, 196.92305849232213, 217.1)]
    toangle_curve(flexdata,angle_parameter)


if __name__ == "__main__":
    #drawSingleValidationbefore("0.txt")
    #DataGenarate()
    #digitGen(100000)
    charGen(100000)
#查看数据分布箱型图
# dataDescription()
# dataPictureDescription()
# digitDataDescription()
# digitDataDescription()
# CQXPictureDataDescription()
# CQXFlexDataDescription()