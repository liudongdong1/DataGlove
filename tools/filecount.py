import os

jsonInfo=[]
def getfileCount(folder):
    for file in os.listdir(folder):
        jsonInfo.append((file,len(os.listdir(os.path.join(folder,file)))))
    print(jsonInfo)

getfileCount(r'../../data/temp/image/word')