目录结构

- util目录函数介绍
  - faceUtil.py: 封装了人脸测试函数，基于开源的face_recognition开源库，识别人脸并绘制人脸landmark和识别姓名信息；
  - flexhelper.py:  封装了弯曲传感器数据读取功能，基于serial串口库进行操作；
  - imghelper.py:  封装摄像头操作，获取每一帧图像数据；
  - uhand.py: 封装了开源套件Uhand ，基于serial串口库进行操作发送控制命令；
- tools目录函数介绍
  - config.py: 系统一些参数介绍
  - filterOp.py: 数据滤波算法，包括平均滑动算法，kalman UKF滤波，低通滤波，KF滤波；
  - video2images.py: 将图片保存为图片；
  - lib_images_io.py： 包含ReadFromFolder； ReadFromVideo； ReadFromWebcam； VideoWriter； ImageDisplayer；
  - image2video.py: 将图片保存为视频文件；
  - skeleton.py: 绘制手部3D关键点；
  - mediapipeTest.py: mediapipe 检测手部关键点测试；
  - model.py: 人手势和下一个字符预测模型介绍；
  - predict.py: RNN预测下一个字符；
  - lib_txtIO.py: 一些文件读取的函数
  - lib_plot.py: 利用matplotlib函数进行一些曲线后的绘制
  - flexQuantify.py:  弯曲传感器数据转化以及一些数据分布查看，测试函数；

- ui_files目录
  - calibration.ui: 电压到弯曲校准函数
  - dataRecord.ui: 采集图片和弯曲传感器数据ui界面
  - create_gesture.ui:  识别手势ui界面
  - hand_control.ui: 利用滑块来控制uhand机械手界面
  - flex_control.ui: 利用手套弯曲传感器来控制uhand机械手
  - FaceRec.ui: 人脸识别登录界面
  - show.ui: 系统首页展示

