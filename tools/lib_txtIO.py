import io
import os
from ast import literal_eval
import numpy as np
# 读文件里面的数据转化为二维列表
def Read_list(filename):
    file1 = open(filename+".txt", "r")
    list_row =file1.readlines()
    list_source = []
    for i in range(len(list_row)):
        column_list = list_row[i].strip().split("\t")  # 每一行split后是一个列表
        list_source.append(column_list)                # 在末尾追加到list_source
    for i in range(len(list_source)):  # 行数
        for j in range(len(list_source[i])):  # 列数
            list_source[i][j]=float(list_source[i][j])
    file1.close()
    return list_source

#保存二维列表到文件
def Save_list(list1,filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')                          # 相当于Tab一下，换一个单元格
        file2.write('\n')                              # 写完一行立马换行
    file2.close()


def numpysave(list1,filename):
    np.savetxt(filename, list1,fmt='%d',delimiter=',')

def numpyload(filename):
    dets= np.loadtxt(filename,delimiter=',')
    return dets

def readData(foler):
    coordinateDataTrain=[]
    tempfile=os.path.join(foler,"angle")
    with open(tempfile) as fileOp:
        lines=fileOp.readlines()            #每个文件长度为1
        #print(len(lines))
        for line in lines:
            final_lrdata = literal_eval(line)
            #print("data",final_lrdata)
            for i in range(0,len(final_lrdata)):
                if(final_lrdata[i][-2]=='Right'):
                    coordinateDataTrain.append([(np.pi-i)*180/np.pi for i in reversed(final_lrdata[i][:-2])])
        return coordinateDataTrain


def readCQXBendData(filename):
    coordinateDataTrain=[]
    data=np.loadtxt(filename,delimiter=',')
    for line in data:
        #print("data",final_lrdata)
        coordinateDataTrain.append([(np.pi-i)*180/np.pi for i in reversed(line)])      
    return coordinateDataTrain

def readCQXPictureData(filename):
    coordinateDataTrain=[]
    with open(filename) as fileOp:
        lines=fileOp.readlines()            #每个文件长度为1
        #print(len(lines))
        for line in lines:
            final_lrdata = literal_eval(line)
            #print("data",final_lrdata)
            for i in range(0,len(final_lrdata)):
                if(final_lrdata[i][-2]=='Right'):
                    coordinateDataTrain.append([(np.pi-i)*180/np.pi for i in reversed(final_lrdata[i][:-2])])
        return coordinateDataTrain

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

def getfileCount(folder):
    '''
        统计数据集个数
    '''
    jsonInfo=[]
    for file in os.listdir(folder):
        jsonInfo.append((file,len(os.listdir(os.path.join(folder,file)))))
    print(jsonInfo)
if __name__ == "__main__":
    # lists=[[1,2,3,4],[45,23,456,23,54,23],[12,23,23,345,23,12]]
    # Save_list(lists,'myfile')
    # print(Read_list('myfile'))
    #data=readData(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\char\B")
    data=readCQXPictureData(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\CQX\pictureAngle\listAlast8.txt")
    print(data[0])
    data1=readCQXBendData(r"D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\data\temp\picFlex\CQX\bendAngle\A.txt")
    print(data1[0])