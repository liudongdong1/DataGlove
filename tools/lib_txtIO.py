import io
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
    np.savetxt(filename, list1,fmt='%f',delimiter=',')

def numpyload(filename):
    dets= np.loadtxt(filename,delimiter=',')
    return dets

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

if __name__ == "__main__":
    lists=[[1,2,3,4],[45,23,456,23,54,23],[12,23,23,345,23,12]]
    Save_list(lists,'myfile')
    print(Read_list('myfile'))