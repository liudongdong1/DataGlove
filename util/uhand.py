import serial
import serial.tools
import serial.tools.list_ports
import binascii
import time


class SerialOp():
    def __init__(self,com,baudrate,timeout):
        self.port = com
        self.baudrate = baudrate
        self.bytesize = 8
        self.stopbits = 1
        self.parity = serial.PARITY_NONE#无校验
        self.timeout =timeout
        global Ret     # flag: 判断是否打开，如果打开，Ret=true
        try:
            # 打开串口，并得到串口对象
            self.ser= serial.Serial(port=self.port,baudrate=self.baudrate,bytesize=self.bytesize,stopbits=self.stopbits,parity=self.parity,timeout=self.timeout)
            # 判断是否打开成功
            if (self.ser.is_open):
               Ret = True
               print("打开串口成功")
            else:
                self.ser.open()
        except Exception as e:
            print("---异常---：", e)

 

    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
        if len(self.Com_Dict) == 0:
            self.state_label.setText(" 无串口")
        else:
            print(self.Com_Dict)

    # 打开串口
    # def port_open(self,port,baudrate,timeout):
    #     self.ser.port = port
    #     self.ser.baudrate = baudrate
    #     self.ser.bytesize = 8
    #     self.ser.stopbits = 1
    #     self.ser.parity = serial.PARITY_NONE#无校验
    #     try:
    #         self.ser.open()
    #         print(self, "Port Error", "此串口能被打开！")
    #     except:
    #         print(self, "Port Error", "此串口不能被打开！")
    #         return None

    def datasend(self,bendvalue):
        '''
            bendvalue为五个传感器数据,float类型, 大拇指id 1，180原来机器180为弯曲状态；二拇指 id：2 等其他原来机器180度为伸直状态 
            bendvalue: 小拇指---》大拇指
        '''
        if len(bendvalue)!=5:
            return None
        hexList=[0x55,0x55,0x14,3,5,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        bendvalue=bendvalue[::-1]   #顺序倒置
        for i in range(0,5):
            hexList[7+i*3]=i+1   #手指 id
            #对大拇指状态进行处理
            if i==0:
                value=int((bendvalue[i]-0)/180*1100+900)
            else: 
                value=int((180-bendvalue[i])/180*1100+900)
            #print(value)
            hexList[7+i*3+1]=int(hex(value & 0x00ff),16)   # 取低八位 
            hexList[7+i*3+2]=int(hex(value >> 8),16)   # 取高八位 
            
#value: 1475 datasend[4]:195 datasend[5]:5
        #print("type",type(hexList),"value",hexList)
        hexList = bytes(hexList)
        #print("type",type(hexList),"value",hexList)
        if self.ser.isOpen():
            num = self.ser.write(hexList)
#UU\x14\x03\x05\x00\x00\x01\xd0\x07\x02\xd0\x07\x03\xd0\x07\x04\xd0\x07\x05\xd0\x07  #str形式显示
#55 55 08 03 01 00 00 02 C3 05  #hex 形式显示

    def datawriteSingleHand(self,index,bendvalue):
        '''
            bendvalue为一个传感器数据,float类型, 大拇指id 1,二拇指 id：2
        '''
        hexList=[0x55,0x55,0x14,3,1,0,0,0,0,0]
        hexList[7]=index  #手指 id
        value=int((bendvalue-0)/180*1100+900)
        hexList[7+1]=int(hex(value & 0x00ff),16)   # 取低八位 
        hexList[7+2]=int(hex(value >> 8),16)   # 取高八位 
#value: 1475 datasend[4]:195 datasend[5]:5
        #print("type",type(hexList),"value",hexList)
        hexList = bytes(hexList)
        #print("type",type(hexList),"value",hexList)
        if self.ser.isOpen():
            num = self.ser.write(hexList)
            

    def data_send(self,input_s):
        if self.ser.isOpen():
            if input_s != "":
                input_s = input_s.strip()
                send_list = []
                while input_s != '':
                    try:
                        num = int(input_s[0:2], 16)  #将一个字符串或数字转换为整型 int(x, base=10)； base进制数16等； int('12',16) 输出18
                    except ValueError:
                        print(self, 'wrong data', '请输入十六进制数据，以空格分开!')
                        return None
                    input_s = input_s[2:].strip()
                    send_list.append(num)
                input_s = bytes(send_list)   # bytes([1,2,3,4])  转化bytes 对象
            num = self.ser.write(input_s)
        else:
            pass

    # 关闭串口
    def port_close(self):
        try:
            self.ser.close()
        except:
            pass

    # 接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()
        except:
            self.port_close()
            return None
        if num > 0:
            data = self.ser.read(num)
            num = len(data)
            print(data)
            f = open("11111111.txt", "a")  # 设置文件对象
            # f.write(float(data[3:10]))
            f.write(str(data)+'\n')
            # 统计接收字符的数量
        # else:
            # print("data null")


if __name__ == '__main__':
    serialOp=SerialOp("COM6", 9600, 0.3)
    a=[0,0,0,0,0]
    while True:
        for i in range(0,5):
            a[i]=(a[i]+10)%180
        serialOp.datasend(a)
        #serialOp.datawriteSingleHand(1,a[0])
        time.sleep(1)
    


