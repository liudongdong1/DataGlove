import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import re
# For static images:


# For webcam input:
def cameraTest():
    cap = cv2.VideoCapture(0)
    print("1111")
    with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


def imageTest(file):
    ''' 
       filename: 待处理手部图片对应数据
    '''
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            return
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imwrite(
            './1.png', cv2.flip(annotated_image, 1))
        with open("./origindatafile.txt",'w') as f1:  #用于存储关键点三维坐标
            f1.write(str(results.multi_hand_landmarks)+'\n')


def video_handTest(file,outputfile):
    import time#用于得知当前时间
    #cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)#捕获摄像头,0一般是笔记本的内置摄像头，1，2，3等等则是接在usb口上的摄像头
    cap = cv2.VideoCapture(file)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    print("视频帧速率",fps)
    mpHands = mp.solutions.hands#简化函数名
    hands = mpHands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5)#配置侦测过程中的相关参数
    mpDraw = mp.solutions.drawing_utils#画点用的函数
    handLmStyle = mpDraw.DrawingSpec(color = (0,0,255),thickness = 5)#点的样式，#线的样式BGR，前一个参数是颜色，后一个是粗细
    handConStyle = mpDraw.DrawingSpec(color = (0,255,0),thickness = 10)#线的样式BGR，#线的样式BGR，前一个参数是颜色，后一个是粗细
    pTime = 0
    cTime = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputfile,fourcc, 20.0, (640,480))
    speed=[]
    while True:#读取视频的循环
        ret,img = cap.read()#读入每一帧图像
        if ret:#如果读取不为空值，则显示画面
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#将BGR图像转化为RGB图像，因为mediapie需要的是RGB
            result = hands.process(imgRGB)#导入图像进行识别
            #print(result.multi_hand_landmarks)
            imgHeight = img.shape[0]#得到图像的高
            imgWeight = img.shape[1]#得到图像的宽
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:#循环一遍所有的坐标
                    mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmStyle,handConStyle)#画出点和线
                    for i,lm in enumerate(handLms.landmark):
                        xPos = int(imgWeight*lm.x)#将坐标转化为整数
                        yPos = int(imgHeight*lm.y)
                        cv2.putText(img,str(i),(xPos-25,yPos+5),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)#将手上对应的点的编号打印在图片上
                        print(i,int(imgHeight*lm.z))#将坐标打印出来
        cTime = time.time()#得到当前时间
        fps = 1/(cTime-pTime)#用1除以播放一帧所用时间就可以得出每秒帧数
        pTime = cTime#得到这一帧结束时的时间
        cv2.putText(img,f"FPS:{int(fps)}",(30,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)#将得到的帧数信息打印在图片上
        cv2.imshow("img", img)#展示图片
        out.write(img)
        if cv2.waitKey(1) ==ord("q"):#如果按下q键，则终止循环
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()



def video_handTest1(file,outputfile):
    import time#用于得知当前时间
    #cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)#捕获摄像头,0一般是笔记本的内置摄像头，1，2，3等等则是接在usb口上的摄像头
    cap = cv2.VideoCapture(file)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    print("视频帧速率",fps)  #30
    mpHands = mp.solutions.hands#简化函数名
    hands = mpHands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5)#配置侦测过程中的相关参数
    mpDraw = mp.solutions.drawing_utils#画点用的函数
    handLmStyle = mpDraw.DrawingSpec(color = (0,0,255),thickness = 5)#点的样式，#线的样式BGR，前一个参数是颜色，后一个是粗细
    handConStyle = mpDraw.DrawingSpec(color = (0,255,0),thickness = 10)#线的样式BGR，#线的样式BGR，前一个参数是颜色，后一个是粗细
    pTime = 0
    cTime = 0
    trajectoryX=[]
    trajectoryY=[]
    trajectoryZ=[]
    ret=True
    while ret:#读取视频的循环
        ret,img = cap.read()#读入每一帧图像
        if ret:#如果读取不为空值，则显示画面
            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#将BGR图像转化为RGB图像，因为mediapie需要的是RGB
            result = hands.process(imgRGB)#导入图像进行识别
            #print(result.multi_hand_landmarks)
            imgHeight = img.shape[0]#得到图像的高
            imgWeight = img.shape[1]#得到图像的宽
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:#循环一遍所有的坐标
                    mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmStyle,handConStyle)#画出点和线
                    for i,lm in enumerate(handLms.landmark):
                        xPos = int(imgWeight*lm.x)#将坐标转化为整数
                        yPos = int(imgHeight*lm.y)
                        zPose=int(imgWeight*lm.z)
                        if i==1:
                            trajectoryX.append(xPos)
                            trajectoryY.append(yPos)
                            trajectoryZ.append(zPose)
                        cv2.putText(img,str(i),(xPos-25,yPos+5),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)#将手上对应的点的编号打印在图片上
        cTime = time.time()#得到当前时间
        fps = 1/(cTime-pTime)#用1除以播放一帧所用时间就可以得出每秒帧数
        pTime = cTime#得到这一帧结束时的时间
        if cv2.waitKey(1) ==ord("q"):#如果按下q键，则终止循环
            break
    import numpy as np
    temp=np.array([trajectoryX,trajectoryY,trajectoryZ])
    np.savetxt("./剪刀.txt", temp,fmt='%d',delimiter=',')
    cap.release()
    cv2.destroyAllWindows()

def hand_record(filename_open, filename_record):  #是得到左右手的数据吗
    '''
        存储每一行是21个点，每一个点有三个坐标xyz 到filename_record文件中, 按顺序存储0(x,y,z), 1(x,y,z)
    '''
    with open(filename_open, 'r') as f:
        f1 = f.read()
        rawdata = re.findall(r"[+-]?[0-9]\.[0-9]+[eE]?[+-]?[0-9]+", f1) 
    print(rawdata)


if __name__ == "__main__":
    import os
    folder=r'C:\Users\liudongdong\OneDrive - tju.edu.cn\桌面\汇报\FlexGlove\data\Video'
    video_handTest(os.path.join(folder,'剪刀.avi'),os.path.join(folder,'视频重复3次IMU_out.avi'))
#hand_record("./origindatafile.txt","1.txt")
# imagefile=r"./hand.png"
# imageTest(imagefile)
#cameraTest()
