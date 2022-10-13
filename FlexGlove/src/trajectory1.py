import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from pykalman import UnscentedKalmanFilter

#filename="../data/IMU/IMUData.csv"
#data = pd.read_csv(filename,usecols=[3,4,5,19,20,21]).values     #取处理后的线性速度x,y,z  [19,20,21]; 加速度 [3，4,5], 使用软件解析的数据

filename="./colectUtil/1.csv"
data = pd.read_csv(filename,usecols=[2,3,4,15,16,17]).values   # 用程序存储的数据解析
t=0.0025               # 根据频率设置的  时间t/s
print(type(data))
print(data[0,:])
#ukf = UnscentedKalmanFilter()

#绘制原始图形
def draw3Line(x,y,z,desp,legend,danwei):
    params={
    'axes.labelsize': '16',       
    'xtick.labelsize':'14',
    'ytick.labelsize':'14',
    'lines.linewidth':2.5 ,
    'legend.fontsize': '14',
    'figure.figsize'   : '12, 8'}
    plt.rcParams.update(params)
    index=[i for i in range(len(x))]
    fig = plt.figure(num=2, figsize=(15, 8),dpi=80)
    plt.title(desp)
    plt.ylabel(danwei)
    p1,=plt.plot(index,x,label="X")
    p2,=plt.plot(index,y,label="Y")
    p3,=plt.plot(index,z,label="Z")
    plt.legend([p1,p2,p3], legend, loc='lower right', scatterpoints=1)
    plt.show()


accx=data[:,0]
accy=data[:,1] 
accz=data[:,2] 
print(accz.shape)
desp="AccelerateX-Y-Z"
lable='m/s^2'
legend=["AccX","AccY","AccZ"]
draw3Line(accx,accy,accz,desp,legend,lable)


speedx=data[:,3]
speedy=data[:,4]
speedz=data[:,5]
desp="SpeedX-Y-Z"
lable='m/s'
legend=["SpeedX","SpeedY","SpeedZ"]
draw3Line(speedx,speedy,speedz,desp,legend,lable)







