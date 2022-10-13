import numpy as np
import os

def DataGenarate():
    basefolder=r"/home/iot/jupyter/root_dir/liudongdong/data/flexData/chars/charFlex/26char"
    savefolder=r"/home/iot/jupyter/root_dir/liudongdong/data/flexData/chars/chartest"
    for folder in os.listdir(basefolder):
        tempdata=[]
        for file in os.listdir(os.path.join(basefolder,folder)):
            print("file:",os.path.join(basefolder,folder,file))
            data=np.loadtxt(os.path.join(basefolder,folder,file),delimiter=',')
            data=np.asarray(data)   # 181*5
            #for i in range(0,5):
            #    data[:,i]=data[:,i]*data[:,i]*param[i][0]+param[i][1]*data[:,i]+param[i][2]+param[i][3]
            if len(tempdata)==0:
                tempdata=data
            else:
                tempdata=np.row_stack((tempdata,data))
        #     tempdata.append(data)
        # tempdata=np.asarray(tempdata)
        print(tempdata.shape)
        np.savetxt(os.path.join(savefolder,"{}.txt".format(folder)),tempdata,delimiter=',',fmt='%d')


def DataGenarateDigit():
    basefolder=r"/home/iot/jupyter/root_dir/liudongdong/data/flexData/digit/digitFlex_7days"
    savefolder=r"/home/iot/jupyter/root_dir/liudongdong/data/flexData/digit/digitTest"
    for folder in os.listdir(basefolder):
        tempdata=[]
        for file in os.listdir(os.path.join(basefolder,folder)):
            print("file:",os.path.join(basefolder,folder,file))
            data=np.loadtxt(os.path.join(basefolder,folder,file),delimiter=',')
            data=np.asarray(data)   # 181*5
            #for i in range(0,5):
            #    data[:,i]=data[:,i]*data[:,i]*param[i][0]+param[i][1]*data[:,i]+param[i][2]+param[i][3]
            if len(tempdata)==0:
                tempdata=data
            else:
                tempdata=np.row_stack((tempdata,data))
        #     tempdata.append(data)
        # tempdata=np.asarray(tempdata)
        print(tempdata.shape)
        np.savetxt(os.path.join(savefolder,"{}.txt".format(folder)),tempdata,delimiter=',',fmt='%d')
        
if __name__ == "__main__":
    DataGenarateDigit()