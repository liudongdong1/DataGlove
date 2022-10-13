# -*- coding: UTF-8 –*-
import cv2
import threading
class VideoCollect(object):
    def __init__(self,cameras):
        self.cap = cv2.VideoCapture(cameras)
        self.camera_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.camera_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.video_width = int(self.camera_width)
        self.video_height = int(self.camera_height)
        # 设置相机宽度
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.video_width)
        # 设置相机高度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.video_height)
        # 设置视频编码，帧率，宽高
        self.video_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.stop=False

    def close(self):
        # 释放摄像头
        self.cap.release()
        
        # 删除全部窗口
        cv2.destroyAllWindows()

    def writeVideo(self,filename):
        video_writer = cv2.VideoWriter(filename, self.video_fourcc,30,(self.video_width,self.video_height))
        print("writeVideo: 存储路径",filename)
        # 判断摄像头是否打开
        print("视频数据采集中...")
        while(self.cap.isOpened() and not self.stop):
            # 从摄像头获取帧数据
            ret, frame = self.cap.read()

            if ret:
                # 显示帧数据
                #cv2.imshow('frame', frame)
                # 向文件中写入帧数据
                video_writer.write(frame)
                # 如果检测到了按键q则退出，不再显示摄像头并且保存视频文件
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break 
        # 释放videowriter
        video_writer.release()

if __name__ == '__main__':
    videocollect=VideoCollect(0)
    filename=r'G:/FlexGlove/data/Video/output1.avi'
    #videocollect.close()
    try:
        thread = threading.Thread(target=videocollect.writeVideo, args=(filename,))
        thread.start()
    except Exception as e:
        print(e)
    print("Input enter to stop recording")
    input()
    videocollect.stop=True
    print("Bye")
    videocollect.close()





