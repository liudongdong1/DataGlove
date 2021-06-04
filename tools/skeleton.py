import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import enum
from ast import literal_eval



# class HandLandmark(enum.IntEnum):
#     """The 21 hand landmarks."""
#     WRIST = 0
#     THUMB_CMC = 1
#     THUMB_MCP = 2
#     THUMB_IP = 3
#     THUMB_TIP = 4
#     INDEX_FINGER_MCP = 5
#     INDEX_FINGER_PIP = 6
#     INDEX_FINGER_DIP = 7
#     INDEX_FINGER_TIP = 8
#     MIDDLE_FINGER_MCP = 9
#     MIDDLE_FINGER_PIP = 10
#     MIDDLE_FINGER_DIP = 11
#     MIDDLE_FINGER_TIP = 12
#     RING_FINGER_MCP = 13
#     RING_FINGER_PIP = 14
#     RING_FINGER_DIP = 15
#     RING_FINGER_TIP = 16
#     PINKY_MCP = 17
#     PINKY_PIP = 18
#     PINKY_DIP = 19
#     PINKY_TIP = 20

# HAND_CONNECTIONS = frozenset([
#     (HandLandmark.WRIST, HandLandmark.THUMB_CMC),         
#     (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),     
#     (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP), 
#     (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP), 
#     (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP), 
#     (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP), 
#     (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP), 
#     (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP), 
#     (HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP), 
#     (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP), 
#     (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP), 
#     (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP), 
#     (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
#     (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP), 
#     (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
#     (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
#     (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP), 
#     (HandLandmark.WRIST, HandLandmark.PINKY_MCP),  
#     (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
#     (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
#     (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP)
# ])

filename=r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\digit\eight\coordinate"
def getData(filename):
    x_data=[]
    y_data=[]
    z_data=[]
    with open(filename, 'r') as fl2:
        lines = fl2.readlines()
        oneRecord = literal_eval(lines[0])
        for i in range(0,len(oneRecord)):
            x_data.append(oneRecord[i][0])
            y_data.append(oneRecord[i][1])
            z_data.append(oneRecord[i][2])
    fl2.close()
    return x_data,y_data,z_data


def getXYZ(a, b):
    return [x_data[a],x_data[b]],[y_data[a],y_data[b]],[z_data[a],z_data[b]]

def draw3Dskeleton(x_data,y_data,z_data):
    #关节点连接情况
    link=[[0,1],[2,3],[1,2],[2,1],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]]

    fig = plt.figure()
    ax = Axes3D(fig)
    #绘制所有地关节点连接信息
    for i in range(len(link)):
        a,b,c=getXYZ(link[i][0],link[i][1])
        #print(a,b,c)
        ax.plot(a,b,c,c='b')

    p1=ax.scatter(x_data,y_data,z_data, c='c',label='Joint Point')  # 绘制数据点,颜色是红色


    #ax.plot_surface(x_data, z_data, z_data, rstride=1, cstride=1, cmap='rainbow')

    ax.set_zlabel('Z(mm)')  # 坐标轴
    ax.set_ylabel('Y(mm)')
    ax.set_xlabel('X(mm)')
    ax.legend([p1],["Joint Point"],loc='lower right', scatterpoints=1)

    plt.draw()
    plt.pause(10)
    plt.show()

if __name__ == "__main__":
    x_data,y_data,z_data=getData(filename)
    draw3Dskeleton(x_data,y_data,z_data)