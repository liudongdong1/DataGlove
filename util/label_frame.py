from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel
import cv2

# QImage *img=new QImage; //新建一个image对象
# img->load(":/new/label/img/wholeBody.jpg"); //将图像资源载入对象img，注意路径，可点进图片右键复制路径
# ui->label1->setPixmap(QPixmap::fromImage(*img)); //将图片放入label，使用setPixmap,注意指针*img
#QLabel 显示一帧图片
class LabelFrame(QLabel):
    def __init__(self, parent=None):
        super(LabelFrame, self).__init__(parent=parent)
        self.main_window = parent
        self.setAlignment(Qt.AlignCenter)  # 居中显示
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: rgb(0, 0, 0);")  # 黑底
        

    #通过该函数实时刷新显示每一帧
    def update_frame(self, frame):
        pixmap = self.img_to_pixmap(frame)
        self.setPixmap(pixmap)
        self.resize_pix_map()

    def resize_pix_map(self):
        """保证图像等比例缩放"""
        pixmap = self.pixmap()
        if not pixmap:
            return
        if self.height() > self.width():
            width = self.width()
            height = int(pixmap.height() * (width / pixmap.width()))
        else:
            height = self.height()
            width = int(pixmap.width() * (height / pixmap.height()))
        pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)

    def resizeEvent(self, *args, **kwargs):
        self.resize_pix_map()

    @staticmethod
    def img_to_pixmap(frame):
        """nparray -> QPixmap"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        h, w, c = frame.shape  # 获取图片形状
        image = QImage(frame, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        return pixmap
