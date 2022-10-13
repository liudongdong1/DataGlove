import sys, os
from collections import OrderedDict
import time
import threading
sys.path.append(r'G:\FlexGlove\lpresearch-lpsensorpy-dad5c37a7475')    # 这里需要修改对应的文件目录
import csv
from lpmslib import LpmsB2
from lpmslib import lputils


TAG="IMUCollection"

class IMUCollect(object):
    def __init__(self,com,baudrate):
        self.com = com
        self.baudrate=baudrate
        self.stop=False
        self.lpmsb = LpmsB2.LpmsB2(self.com, self.baudrate)
    
    def connect(self):
        lputils.logd(TAG, "Connecting sensor")
        if self.lpmsb.connect():
            lputils.logd(TAG, "Connected")
            config_reg = self.lpmsb.get_config_register()
            print(config_reg)

    def disconnect_sensor(self):
        lputils.logd(TAG, "Disconnecting sensor")
        self.lpmsb.disconnect()
        lputils.logd(TAG, "Disconnected")


    def get_stream_data(self):
        sensor_data = self.lpmsb.get_stream_data()
        self.pretty_print_sensor_data(sensor_data)

    def get_sensor_data():
        sensor_data = self.lpmsb.get_sensor_data()
        self.pretty_print_sensor_data(sensor_data)

    def savetoCSV(self,filename):
        if not self.lpmsb.connect():
            self.connect()
        with open(filename, 'w',newline='') as f:
            myWriter = csv.writer(f, quoting=csv.QUOTE_ALL)
            #SensorId, TimeStamp (s), FrameNumber, AccX (g), AccY (g), AccZ (g), GyroX (deg/s), GyroY (deg/s), GyroZ (deg/s), MagX (uT), MagY (uT), MagZ (uT), EulerX (deg), EulerY (deg), EulerZ (deg), QuatW, QuatX, QuatY, QuatZ, LinAccX (g), LinAccY (g), LinAccZ (g), Pressure (kPa), Altitude (m), Temperature (degC), HeaveMotion (m)
            myWriter.writerow(['TimeStamp', 'FrameCounter', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ',
                                'QuatW', 'QuatX', 'QuatY', 'QuatZ', 'EulerX', 'EulerY', 'EulerZ', 'LinAccX',
                                'LinAccY', 'LinAccZ'])
            
            # Set 16bit data to reduce amount of stream datpa
            self.lpmsb.set_16bit_mode()
            
            # Set stream frequency
            self.lpmsb.set_stream_frequency_200Hz()
            self.lpmsb.clear_data_queue()
            while not self.stop:
                sensor_data = self.lpmsb.get_stream_data()
                if not sensor_data:
                    continue

                data = [sensor_data[1]*0.0025, 
                        sensor_data[2],
                        sensor_data[6][0], sensor_data[6][1], sensor_data[6][2],
                        sensor_data[7][0], sensor_data[7][1], sensor_data[7][2],
                        sensor_data[9][0], sensor_data[9][1], sensor_data[9][2], sensor_data[9][3],
                        sensor_data[10][0], sensor_data[10][1], sensor_data[10][2],
                        sensor_data[11][0], sensor_data[11][1], sensor_data[11][2]
                        ]
                myWriter.writerow(data)
                #print(type(data),data)
            self.lpmsb.disconnect()

    def pretty_print_sensor_data(self,sensor_data):
        j = 25
        d = '.'
        print ("IMU ID:".ljust(j, d), sensor_data[0])
        print ("TimeStamp(s):".ljust(j, d), sensor_data[1]*0.0025)
        print ("Frame Counter:".ljust(j, d), sensor_data[2])
        print ("Battery Level:".ljust(j, d), sensor_data[3])
        print ("Battery Voltage:".ljust(j, d), sensor_data[4])
        print ("Temperature:".ljust(j, d), sensor_data[5])
        print ("Acc:".ljust(j, d), ['%+.3f' % f for f in sensor_data[6]])
        print ("Gyr:".ljust(j, d), ['%+.3f' % f for f in sensor_data[7]])
        print ("Mag:".ljust(j, d), ['%+.3f' % f for f in sensor_data[8]])
        print ("Quat:".ljust(j, d), ['%+.3f' % f for f in sensor_data[9]])
        print ("Euler:".ljust(j, d), ['%+.3f' % f for f in sensor_data[10]])
        print ("LinAcc:".ljust(j, d), ['%+.3f' % f for f in sensor_data[11]])


if __name__ == '__main__':
    imuCollect=IMUCollect("COM5",115200 )
    imuCollect.connect()
    imuCollect.get_stream_data()
    try:
        thread = threading.Thread(target=imuCollect.savetoCSV, args=("1.csv",))
        thread.start()
    except Exception as e:
        print(e)
    print("Input enter to stop recording")
    input()
    imuCollect.stop=True
    print("Bye")
    

