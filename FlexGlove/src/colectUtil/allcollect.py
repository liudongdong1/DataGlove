from flexCollect import FlexSensor
from videoCollect import VideoCollect
from imuCollect import IMUCollect
import threading
import os
class CollectEntry(object):
    def __init__(self,imuport,baudrate,flexport,video):
        self.imu_collect=IMUCollect(imuport,baudrate)
        self.videocollect=VideoCollect(video)
    
    def connect(self):
        self.imu_collect.connect()
        pass

    def record(self,imufile,cvfile):
        try:
            threadimu = threading.Thread(target=self.imu_collect.savetoCSV, args=(imufile,))
            threadcv = threading.Thread(target=self.videocollect.writeVideo, args=(cvfile,))
            threadimu.start()
            threadcv.start()
        except Exception as e:
            print(e)

    def enableRecord(self):
        self.imu_collect.stop=False
        self.videocollect.stop=False

    def endrecord(self):
        self.imu_collect.stop=True
        self.videocollect.stop=True


if __name__=='__main__':
    collectEntry=CollectEntry("COM5",115200,"COM6",0)
    collectEntry.connect()
    
    folder=r'G:\FlexGlove\data'
    filename="record1"
    imufile=os.path.join(folder,"IMU",filename+".csv")
    cvfile=os.path.join(folder,"Video",filename+".avi")
    collectEntry.enableRecord()
    print("Input enter to start recording")
    input()
    print("Input enter to stop recording")
    collectEntry.record(imufile,cvfile)
    
    input()
    collectEntry.endrecord()
    print("Bye")

