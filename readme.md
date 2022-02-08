- 视频地址：https://www.bilibili.com/video/BV14Z4y1R72A?share_source=copy_web

![人脸认证登录](https://github.com/liudongdong1/DataGlove/raw/main/icons/%E4%BA%BA%E8%84%B8%E8%AE%A4%E8%AF%81%E7%99%BB%E5%BD%95.jpg)

<iframe src="//player.bilibili.com/player.html?aid=381461982&bvid=BV14Z4y1R72A&cid=504190104&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

目录结构

- kafka目录： 是使用kafka 消息传递机制的测试
- data目录：
  - validation: 存储每次矫正一个完整弯曲过程的电压数据
  - DatasetAlpha: A-Z对应的手部图片数据，其中没有J,Z
  - face: 人脸图片数据
  - flexSensor: 程序运行时弯曲传感器数据保存的目录
  - image: 程序运行时图片数据保存的目录
  - rnndata: 训练rnn，下一个单词预测的数据集
  - temp：
    - SampleGesture: 样本数据集，每一个label对应一张图片
    - picFlex： 
    - image目录： 
      - annotated_image: 运行mediapipe后手部标有关键点的图片数据
      - digit: 1-10运行mediapipe后手部标有关键点的图片数据
      - word: 手势字符运行mediapipe后手部标有关键点的图片数据
- outputpic: 目录介绍
  - pptpng: 是一些ppt上图片数据
  - validation: 是矫正结果对应的图片数据
  - validation.txt: 矫正结果对应的原始电压数据
  - train.txt: 是rnn文本预测对应的训练数据集
- util目录函数介绍
  - faceUtil.py: 封装了人脸测试函数，基于开源的face_recognition开源库，识别人脸并绘制人脸landmark和识别姓名信息；
  - flexhelper.py:  封装了弯曲传感器数据读取功能，基于serial串口库进行操作；
  - imghelper.py:  封装摄像头操作，获取每一帧图像数据；
  - uhand.py: 封装了开源套件Uhand ，基于serial串口库进行操作发送控制命令；
  - coordi2angle.py: 手部坐标数据转化为 弯曲角度函数
  - pictransfer.py: 利用图片数据转化对应的弯曲传感器数据角度
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

- main.py: 主程序入口
- FlexDataValidation.py: 测试文件，弯曲传感器矫正测试
- result.txt: 识别输出结果
