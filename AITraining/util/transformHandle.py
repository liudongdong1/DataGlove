import numpy as np

def fivePoint(data):
    temp=[]
    for i in range(0,len(data)):
        temp.append(data[i])
        for j in range(0,len(data)):
            if i!=j:
                temp.append(data[i]-data[j])    #这个有正负值
    data=np.array(temp)
    data=np.reshape(data,(5,5))    #直接data.reshape() 不起作用
    return data
    
def fivePointArray(data):
    temp=[]
    for i in range(0,len(data)):
        temp.append(data[i])
        for j in range(0,len(data)):
            if i!=j:
                temp.append(data[i]-data[j])    #这个有正负值
    data=np.array(temp)
    #data=np.reshape(data,(5,5))    #直接data.reshape() 不起作用
    return data


def fivePointAdj(data):
    data=standardization(data)
    temp=[]
    for i in range(0,len(data)):
        temp.append(data[i])
        for j in range(0,len(data)):
            if i!=j:
                if data[i]-data[j]>0:
                    temp.append(data[i]-data[j])   #只有正直，负值用0表示
                else:
                    temp.append(0)
    data=np.array(temp)
    data=np.reshape(data,(5,5))    #直接data.reshape() 不起作用
    return data
        
def fivePointAdjArray(data):
    data=standardization(data)
    temp=[]
    for i in range(0,len(data)):
        temp.append(data[i])
        for j in range(0,len(data)):
            if i!=j:
                if data[i]-data[j]>0:
                    temp.append(data[i]-data[j])   #只有正直，负值用0表示
                else:
                    temp.append(0)
    data=np.array(temp)
    #data=np.reshape(data,(1，))    #直接data.reshape() 不起作用
    return data


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def sixPoint(data):
    meanvalue=np.min(data,axis=0)
    data=np.append(data,meanvalue)
    data=standardization(data)
    temp=[]
    for i in range(0,len(data)):
        temp.append(data[i])
        for j in range(0,len(data)):
            if i!=j:
                temp.append(data[i]-data[j])   #只有正直，负值用0表示
    data=np.array(temp)
    data=np.reshape(data,(6,6))    #直接data.reshape() 不起作用
    return data

# 邻接矩阵，其中负值 设置为0
def sixPointAdj(data):
    meanvalue=np.min(data,axis=0)
    data=np.append(data,meanvalue)
    data=standardization(data)
    temp=[]
    for i in range(0,len(data)):
        temp.append(data[i])
        for j in range(0,len(data)):
            if i!=j:
                if data[i]-data[j]>0:
                    temp.append(data[i]-data[j])   #只有正直，负值用0表示
                else:
                    temp.append(0)
    adj=np.array(temp)
    adj=np.reshape(adj,(6,6))    #直接data.reshape() 不起作用
    #data=np.reshape(data,(1,6))
    return adj
