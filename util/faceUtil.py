import face_recognition
import cv2
import numpy as np
import os
import time
from threading import Thread
from PyQt5.QtCore import *

from PIL import Image, ImageDraw, ImageFont



class Face_Recognizer:
    def __init__(self, basefolder):
        self.basefolder=basefolder
        self.faces,self.faceNames=self.initFaceData()
    
    '''
    @function: 初始化人脸数据，从已经存储的文件加载name和对应的人脸 encoding
    '''
    def initFaceData(self):
        known_faces=[]
        known_faceNames=[]
        for file in os.listdir(self.basefolder):
            filepath=os.path.join(self.basefolder,file)
            #print(filepath)
            image=face_recognition.load_image_file(filepath)
            try:
                imageEncoding=face_recognition.face_encodings(image)[0]
                known_faceNames.append(file.split('.')[0])
                known_faces.append(imageEncoding)
                print(known_faceNames[-1])
            except:
                print("file don't detect face")
        return known_faces,known_faceNames

    #This function will take a sample frame
    #save the picture of the given user in a folder
    #returns the path of the saved image
    def saveFaceImage(self,imgdata, face_name='user'):
        face_name = '{}.png'.format(face_name)
        facesavepath=os.path.join(self.basefolder, face_name)
        try:
            cv2.imwrite(facesavepath, imgdata)
            print("saveFaceImage OK,name=",face_name,"path=",facesavepath)
        except:
            print("Can't Save File")
    
    '''
    @function: 如果图片数据中包含人脸，则添加人脸的编码和name信息
    @parameters：
        imgdata: 人脸数据
        face_names: 名称数据
    @return
        true: 录入人脸信息成功
        false: 录入人脸信息失败
    '''
    def faceRegister(self,originimage,face_name):
        #imgdata = face_recognition.load_image_file(self.face_image_path)
        # if face_name not in self.faceNames:
        #     imgdata = cv2.resize(originimage, (0, 0), fx=0.25, fy=0.25)
        #     face_encoding = face_recognition.face_encodings(imgdata)[0]
        #     self.faces.append(face_encoding)
        #     self.faceNames.append(face_name)
        #     print("faceRegister: add facecoding ok")
        # self.saveFaceImage(originimage,face_name)
        # print("faceRegister: save face ok")
        # return True
        try:
            if face_name not in self.faceNames:
                imgdata = cv2.resize(originimage, (0, 0), fx=0.25, fy=0.25)
                face_encoding = face_recognition.face_encodings(imgdata)[0]
                self.faces.append(face_encoding)
                self.faceNames.append(face_name)
                print("faceRegister: add facecoding ok")
            self.saveFaceImage(originimage,face_name)
            print("faceRegister: save face ok")
            return True
        except Exception as err:
            print("No face found in the image",err)
            return False



    '''
        @parameter: face_name: 用户名称
        @return: 如果列表包含用户名称，则返回 true； 否则返回false；
    '''
    def nameContain(self, face_name):
        registered = False
        namelist=os.listdir(self.basefolder)
        if '{}.png'.format(face_name) in namelist:
            registered=True
        return registered

    '''
    @function: 从一张图片中识别人脸信息
    @parameters: 
        targetPath: 待识别的图片路径
    @returns: 识别用户的姓名信息
    '''
    def getFaceNameFromFile(self,targetPath):
        name="None"
        image=face_recognition.load_image_file(targetPath)
        try:
            face_encoding=face_recognition.face_encodings(image)[0]
            matches = face_recognition.compare_faces(self.faces, face_encoding)
            face_distances = face_recognition.face_distance(self.faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            #print("最小距离： ",face_distances[best_match_index])
            #print(face_distances)
            if matches[best_match_index]:
                name = self.faceNames[best_match_index]
            return name
        except Exception as e:
            print("file don't detect face",e)
            return name   
    
    '''
    @function: 从一张图片编码中识别人脸信息
    @parameters: 
        targetEncoding: 待识别的图片 特征编码
    @returns: 识别用户的姓名信息
    '''
    def getFaceNameFromEncoding(self,targetEncoding):
        name="None"
        matches = face_recognition.compare_faces(self.faces, targetEncoding)
        face_distances = face_recognition.face_distance(self.faces, targetEncoding)
        best_match_index = np.argmin(face_distances)
        print("最小距离： ",face_distances[best_match_index])
        print(face_distances)
        if matches[best_match_index]:
            name = self.faceNames[best_match_index]
        return name

    def compareToDatabase(self, unknown_face_encoding=None):
        if not self.is_running:
            self.is_running = True
            self.m_thread = Thread(target= self._compareToDatabase )
            self.m_thread.start()

    def _compareToDatabase(self,originimage):
        authenticated = False
        #imgdata = cv2.resize(originimage, (0, 0), fx=0.25, fy=0.25)
        #imgdata=originimage
        #print(type(originimage))
        #small_frame = cv2.resize(originimage, (0, 0), fx=0.25, fy=0.25)
        #rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
        rgb_small_frame=originimage
        time1=time.time()
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
        time2=time.time()
        print("facelandmarks timecost:",time2-time1)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        time3=time.time()
        print("face_locations timecost:",time3-time2)
        print("_compareToDatabase1")
        if(len(face_locations)>1):
            return "Unknown"
        encodings=face_recognition.face_encodings(rgb_small_frame, face_locations)
        time4=time.time()
        print("encodings timecost:",time4-time3)
        #print(type(encodings),len(encodings))
        face_encoding = encodings[0]

        #print(type(self.faces),type(face_encoding))
        matches = face_recognition.compare_faces(self.faces, face_encoding)
        face_distances = face_recognition.face_distance(self.faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        #print("最小距离： ",face_distances[best_match_index])
        #print(face_distances)
        if matches[best_match_index]:
            name = self.faceNames[best_match_index]
        print("_compareToDataset",name,face_locations)

        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                for pt_pos in face_landmarks[facial_feature]:
                    cv2.circle(originimage,pt_pos, 1, (255, 0, 0), 2)
                    cv2.circle(originimage,pt_pos, 5, color=(0, 255, 0))
                        #cv2.circle(originimage, (pt_pos[0]*4,pt_pos[1]*4), 1, (255, 0, 0), 2)
                        #cv2.circle(originimage, (pt_pos[0]*4,pt_pos[1]*4), 5, color=(0, 255, 0))
            
        #process_this_frame = not process_this_frame
        top, right, bottom, left=face_locations[0]
        # top=top*4
        # right=right*4
        # bottom=bottom*4
        # left=left*4
        # Draw a box around the face
        cv2.rectangle(originimage, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        #cv2.rectangle(originimage, (left-20, bottom - 60), (right+20, bottom+20), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        img_PIL = Image.fromarray(originimage)
        font = ImageFont.truetype(r'D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\src\icons\方正康体简体.TTF', 40)
        # 字体颜色
        fillColor = (255,0,0)
        # 文字输出位置
        position = (left - 40, bottom +40)
        textinfo = "欢迎{}登录".format(name)
        # 需要先把输出的中文字符转换成Unicode编码形式
        if not isinstance(textinfo, str):
            textinfo = textinfo.decode('utf8')
    
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position, textinfo, font=font, fill=fillColor)
        # 使用PIL中的save方法保存图片到本地
        # img_PIL.save('02.jpg', 'jpeg')
        # 转换回OpenCV格式
        originimage = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

       # cv2.putText(originimage,"欢迎{}登录".format(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imwrite("tmp.png" , originimage)
        #self.im_s.new_image.emit(".tmp.png")
            # Display the resulting image
        #cv2.imshow('Video', imgdata)
        return originimage,name 


    def removeFaceData(self, face_name):
        pass

    def paint_chinese_opencv(self,im,chinese,pos,color):
        img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype('NotoSansCJK-Bold.ttc',25)
        fillColor = color #(255,0,0)
        position = pos #(100,100)
        if not isinstance(chinese,str):
            chinese = chinese.decode('utf-8')
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position,chinese,font=font,fill=fillColor)

        img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
        return img
#编写一个测试
def functionTest():
    capture = cv2.VideoCapture(0)
    faceServer=Face_Recognizer("../../data/face")
    i=0
    name="liudongdong"
    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame,1)   #镜像操作
        
        key = cv2.waitKey(50)
        #print(key)
        if key  == ord('q'):  #判断是哪一个键按下
            if faceServer.faceRegister(frame,name):
                print("人脸信息录入成功")
            else: print("人脸信息录入失败")
        if key == ord('r'):
            frame,name=faceServer._compareToDatabase(frame)
            print("识别名称，",name)
        cv2.imshow("video", frame)

        #cv2.imshow('Video', imgdata)
        if key== ord('b'):
            break
        
    cv2.destroyAllWindows()
def testFromImage(path): 
    
    faceServer=Face_Recognizer("../../data/face")
    print(faceServer.getFaceNameFromFile(path))
    image=face_recognition.load_image_file(path)
    frame,name=faceServer._compareToDatabase(image)
    print("识别名称，",name)
    cv2.imwrite("video.png", frame)
if __name__ == '__main__':
    #functionTest()

    testFromImage(r'D:\work_OneNote\OneDrive - tju.edu.cn\文档\work_组会比赛\数据手套\DashBoard\src\2.png')
    
    # while i<50:
    #     i=i+1
    #     ret, frame = video_capture.read()
    #     cv2.imshow('Video', frame)
    #     if i<10:
    #         if faceServer.faceRegister(frame,name):
    #             print("人脸信息录入成功")
    #         else: print("人脸信息录入失败")
    #     else:
    #         imgdata,name=faceServer._compareToDatabase(frame)
    #         cv2.imshow('Video', imgdata)
    #         print("识别姓名： name=",name)



'''
recognizer = Face_Recognizer()
recognizer.registerFace()
recognizer.saveFaceImage('Kareem')
'''

