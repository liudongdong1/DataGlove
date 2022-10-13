import sys, os
from collections import OrderedDict
from datetime import datetime
import time
import threading
import multiprocessing
import csv
#import keyboard

from lpmslib import LpmsB2
from lpmslib import lputils

TAG = "MAIN"

def readSensorData2CSV(sensorIDstr, port, Global): 
    TAG = sensorIDstr
    baudrate = 921600
        
       
    # Connect to sensor
    sensor = LpmsB2.LpmsB2(port, Global['baudrate'])
    lputils.logd(TAG, "Connecting sensor " + sensorIDstr)
    if not sensor.connect():
        return

    lputils.logd(TAG, "Connected")

    with open(''.join([Global['FilePrefix'], sensorIDstr, '.csv']), 'wb') as f:
        myWriter = csv.writer(f, quoting=csv.QUOTE_ALL)
        myWriter.writerow(['TimeStamp', 'FrameCounter', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ',
                               'QuatW', 'QuatX', 'QuatY', 'QuatZ', 'EulerX', 'EulerY', 'EulerZ', 'LinAccX',
                               'LinAccY', 'LinAccZ'])
        
        # Set 16bit data to reduce amount of stream datpa
        sensor.set_16bit_mode()
        
        # Set stream frequency
        sensor.set_stream_frequency_200Hz()

        # Put sensor in sync mode 
        sensor.start_sync()
        time.sleep(1)

        # Clear sensor internal data queue
        sensor.clear_data_queue()

        # Sensor ready
        Global[sensorIDstr] = True

        # Wait for sync signal
        while not Global['stopSync']:
            continue
        sensor.stop_sync()


        while not Global['quit']:
            sensor_data = sensor.get_stream_data()
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
            
        sensor.disconnect()

        lputils.logd(TAG, "Terminated")
   

def elapsedTimePrinter(Global):
    start_time = time.time()
    while not Global['quit']:

        elapsed_time = time.time() - start_time
        print "\rElapsed time(s): %f\t"%(elapsed_time) ,
        time.sleep(.2)


def main():
    # Settings
    port1 = 'COM24'
    port2 = 'COM64'
    baudrate = 921600
    sensor1Id = 'lpms1'
    sensor2Id = 'lpms2'
    dateTime = datetime.now().strftime("%Y%m%d_%H%M%S_")

    manager = multiprocessing.Manager()
    Global = manager.dict()

    Global['quit'] = False
    Global['stopSync'] = False
    Global['baudrate'] = baudrate
    Global['FilePrefix'] = dateTime
    Global[sensor1Id] = False       # Sensor ready flag
    Global[sensor2Id] = False       # Sensor ready flag

    # Start sensor thread
    p1 = multiprocessing.Process(target=readSensorData2CSV, args=(sensor1Id, port1, Global))
    p2 = multiprocessing.Process(target=readSensorData2CSV, args=(sensor2Id, port2, Global))
    p3 = multiprocessing.Process(target=elapsedTimePrinter, args=(Global,))

    p1.start()
    p2.start()

    # Wait for sensors to connect
    print("Waiting for sensor to connect")
    while not Global[sensor1Id] and not Global[sensor2Id]:
        time.sleep(2)

    # Sync and start data logging
    raw_input("Sensors connected, Input enter to sync and start recording process")  
    Global['stopSync'] = True
    print("Input enter to stop recording")
    p3.start();

    # Wait for quit command
    raw_input("")

    Global['quit'] = True

    p1.join()
    p2.join()
    p3.join()
    print("Bye")

if __name__ == "__main__":
    main()