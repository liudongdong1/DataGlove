import pandas as pd
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from pykalman import UnscentedKalmanFilter

data = pd.read_csv("../data/IMU/IMUData.csv",usecols=[19,20,21])     #取处理后的线性速度x,y,z
data=data*9.8       #加速度单位  m/s2
t=0.0025               # 根据频率设置的  时间t/s

ukf=UnscentedKalmanFilter(transition_covariance=0.01,observation_covariance=0.1)
#ukf = UnscentedKalmanFilter()

accx=[]               #存储原始加速度
accy=[]
accz=[]
timestamp = []

ChangeVX=[]    #存出速度变化量
ChangeVY=[]
ChangeVZ=[]

low_thre = 0.25
high_thre = 2
for i in range(0,len(data)):                        #每一个采样之间的速度变化量
    if abs(data.iloc[i][0])<low_thre or abs(data.iloc[i][0])>high_thre:
        data.iloc[i][0]=0
    if abs(data.iloc[i][1])<low_thre or abs(data.iloc[i][1])>high_thre:
        data.iloc[i][1]=0
    if abs(data.iloc[i][2])<low_thre or abs(data.iloc[i][2])>high_thre:
        data.iloc[i][2]=0   
    accx.append(data.iloc[i][0])
    accy.append(data.iloc[i][1])
    accz.append(data.iloc[i][2])
    timestamp.append(t*i)

d = 10
for i in range(0, 2*d):
    accx[i] = 0
    accy[i] = 0
    accz[i] = 0
for i in range(len(accx)-2*d, len(accx)):
    accx[i] = 0
    accy[i] = 0
    accz[i] = 0

#绘制原始加速度图
# index=[i for i in range(len(accx))]
# fig = plt.figure(num=2, figsize=(15, 8),dpi=80)
# plt.title("AccelerateX-Y-Z")
# plt.ylabel('m/s^2')
# p1,=plt.plot(index,accx,label="ChangeVX")
# p2,=plt.plot(index,accy,label="ChangeVY")
# p3,=plt.plot(index,accz,label="ChangeVZ")
# plt.legend([p1,p2,p3], ["AccX","AccY","AccZ"], loc='lower right', scatterpoints=1)
# plt.show()

#一个点前后d个点都是0，也置为0
for i in range(d, len(accx)-d-1):
    sumx = 0
    sumy = 0
    sumz = 0
    for j in range(i-d, i+d):
        sumx += accx[j]
        sumy += accy[j]
        sumz += accz[j]

    if sumx==accx[i] or sumy==accy[i] or sumz==accz[i]:
        accx[i] = 0
        accy[i] = 0
        accz[i] = 0



index=[i for i in range(len(accx))]
fig = plt.figure(num=2, figsize=(15, 8),dpi=80)
plt.title("AccelerateX-Y-Z")
plt.ylabel('m/s^2')
p1,=plt.plot(index,accx,label="ChangeVX")
p2,=plt.plot(index,accy,label="ChangeVY")
p3,=plt.plot(index,accz,label="ChangeVZ")
plt.legend([p1,p2,p3], ["AccX","AccY","AccZ"], loc='lower right', scatterpoints=1)
plt.show()


#低通滤波
b, a = signal.butter(4, 0.02, 'lowpass') 
xdata = signal.filtfilt(b, a, accx)
ydata = signal.filtfilt(b, a, accy)
zdata = signal.filtfilt(b, a, accz)

# index=[i for i in range(len(xdata))]
# fig = plt.figure(num=2, figsize=(15, 8),dpi=80)
# plt.title("AccelerateX-Y-Z-Lowpass")
# plt.ylabel('m/s^2')
# p1,=plt.plot(index,xdata,label="ChangeVX")
# p2,=plt.plot(index,ydata,label="ChangeVY")
# p3,=plt.plot(index,zdata,label="ChangeVZ")
# plt.legend([p1,p2,p3], ["AccX","AccY","AccZ"], loc='lower right', scatterpoints=1)
# plt.show()


#EWMA
a = 0.02
coefficient = []
denominator = 0
d = 15
for i in range(0, d):
    coefficient.append(math.pow(1-a, i))
    denominator += math.pow(1-a, i)

for i in range(d, len(xdata)):
    ex = 0
    for j in range(0, d):
        ex += xdata[i-j]*coefficient[j]
    xdata[i] = round(ex/denominator, 2)

    ey = 0
    for j in range(0, d):
        ey += ydata[i-j]*coefficient[j]
    ydata[i] = round(ey/denominator, 2)

    ez = 0
    for j in range(0, d):
        ez += zdata[i-j]*coefficient[j]
    zdata[i] = round(ez/denominator, 2)

index=[i for i in range(len(xdata))]
fig = plt.figure(num=2, figsize=(15, 8),dpi=80)
plt.title("AccelerateX-Y-Z-EWMA")
plt.ylabel('m/s^2')
p1,=plt.plot(index,xdata,label="ChangeAX", c='r')
p2,=plt.plot(index,ydata,label="ChangeAY", c='g')
p3,=plt.plot(index,zdata,label="ChangeAZ", c='b')
plt.legend([p1,p2,p3], ["AccX","AccY","AccZ"], loc='lower right', scatterpoints=1)
plt.show()

#根据方差选开始和结束时间
d = 15
cur = []
for i in range(0, d):
    cur.append(xdata[i])

curvature = []
for k in range(d, len(xdata)-1):
    cur_var = np.var(cur, ddof = 1)
    curvature.append(cur_var*20)
    del cur[0]
    cur.append(xdata[k])

# plt.plot(range(0, len(curvature)), curvature, c='r')
# plt.show()

start_index = 0
end_index = len(curvature)-1
while curvature[start_index]<0.0002:
    start_index += 1

while curvature[end_index]<0.0002:
    end_index -= 1

#plt.scatter(start_index, curvature[start_index], marker='o', c='b')
#plt.scatter(end_index, curvature[end_index], marker='o', c='b')
#plt.scatter(range(0, len(curvature)), curvature, marker='.', c='r')
#plt.scatter(range(0, len(xdata)), xdata, marker='.', c='r')
#plt.show()

#积分求速度
ChangeTime = []
# d = 20
#start_index = 0
#end_index = len(xdata)-1
for i in range(start_index, end_index):
    ChangeVX.append(np.trapz(xdata[start_index:i], timestamp[start_index:i]))
    ChangeVY.append(np.trapz(ydata[start_index:i], timestamp[start_index:i]))
    ChangeVZ.append(np.trapz(zdata[start_index:i], timestamp[start_index:i]))
    ChangeTime.append((i-start_index)*t)

#handwriting论文的矫正速度方法
start_index = 0
end_index = len(ChangeVX)-1
offx = (ChangeVX[end_index]-ChangeVX[end_index-400])/(ChangeTime[end_index]-ChangeTime[start_index])
offy = (ChangeVY[end_index]-ChangeVY[end_index-400])/(ChangeTime[end_index]-ChangeTime[start_index])
offz = (ChangeVZ[end_index]-ChangeVZ[end_index-400])/(ChangeTime[end_index]-ChangeTime[start_index])
for i in range(0, len(ChangeVX)):
    ChangeVX[i] = ChangeVX[i] + (ChangeTime[i]-ChangeTime[start_index]+1)*offx
    ChangeVY[i] = ChangeVY[i] + (ChangeTime[i]-ChangeTime[start_index]+1)*offy
    ChangeVZ[i] = ChangeVZ[i] + (ChangeTime[i]-ChangeTime[start_index]+1)*offz

#末速度为0的矫正方法
lenth = len(ChangeVX)
bx1 = (ChangeVX[0] - ChangeVX[lenth-1])/ChangeTime[lenth-1]
bx2 = -ChangeVX[0]
by1 = (ChangeVY[0] - ChangeVY[lenth-1])/ChangeTime[lenth-1]
by2 = -ChangeVY[0]
bz1 = (ChangeVZ[0] - ChangeVZ[lenth-1])/ChangeTime[lenth-1]
bz2 = -ChangeVZ[0]
for i in range(0, len(ChangeVX)):
    ChangeVX[i] = ChangeVX[i] + bx1*ChangeTime[i]+bx2
    ChangeVY[i] = ChangeVY[i] + by1*ChangeTime[i]+by2
    ChangeVZ[i] = ChangeVZ[i] + bz1*ChangeTime[i]+bz2

#绘制原始每个点速度变化量图
index=[i for i in range(len(ChangeVX))]
fig = plt.figure(num=2, figsize=(15, 8),dpi=80)
plt.title("ChangeVelocityX-Y-Z change")
plt.ylabel('m/s')
p1,=plt.plot(index,ChangeVX,label="ChangeVX")
p2,=plt.plot(index,ChangeVY,label="ChangeVY")
p3,=plt.plot(index,ChangeVZ,label="ChangeVZ")
plt.legend([p1,p2,p3], ["VchgX","VchgY","VchgZ"], loc='lower right', scatterpoints=1)
plt.show()


#积分求位移
pointX=[]    
pointY=[]
pointZ=[]
for i in range(0, len(ChangeVX)):
    pointX.append(np.trapz(ChangeVX[0:i], timestamp[0:i]))
    pointY.append(np.trapz(ChangeVY[0:i], timestamp[0:i]))
    pointZ.append(np.trapz(ChangeVZ[0:i], timestamp[0:i]))

#末速度为0的方法矫正位移
lenth = len(pointX)
bx1 = (pointX[0] - pointX[lenth-1])/ChangeTime[lenth-1]
bx2 = -pointX[0]
by1 = (pointY[0] - pointY[lenth-1])/ChangeTime[lenth-1]
by2 = -pointY[0]
bz1 = (pointZ[0] - pointZ[lenth-1])/ChangeTime[lenth-1]
bz2 = -pointZ[0]
for i in range(0, len(pointX)):
    pointX[i] = pointX[i] + bx1*ChangeTime[i] + bx2
    pointY[i] = pointY[i] + by1*ChangeTime[i] + by2
    pointZ[i] = pointZ[i] + bz1*ChangeTime[i] + bz2

index=[i for i in range(len(pointX))]
fig = plt.figure(num=2, figsize=(15, 8),dpi=80)
plt.title("Index X-Y-Z change")
plt.ylabel('m')
p1,=plt.plot(index,pointX,label="pointVX")
p2,=plt.plot(index,pointY,label="pointVY")
p3,=plt.plot(index,pointZ,label="pointVZ")
plt.legend([p1,p2,p3], ["X","Y","Z"], loc='lower right', scatterpoints=1)
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
p1=ax.plot(pointX, pointY, pointZ,label='Joint Point')  # 绘制数据点,颜色是红色
#ax.plot_surface(x_data, z_data, z_data, rstride=1, cstride=1, cmap='rainbow')
ax.set_zlabel('Z/m')  # 坐标轴
ax.set_ylabel('Y/m')
ax.set_xlabel('X/m')
ax.legend([p1],["Trajectory"],loc='lower right', scatterpoints=1)
plt.show()