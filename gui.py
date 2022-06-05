import sys
from PyQt5.QtWidgets import (QWidget, QMainWindow, QToolTip, QDesktopWidget, QFileDialog, QTextEdit, QAction,
                             QPushButton, QApplication, QLabel, QStackedLayout, QCalendarWidget, QProgressBar)
from PyQt5.QtGui import QFont, QIcon, QBrush, QPainter, QColor, QPixmap
from PyQt5.QtCore import Qt, QDate
from PyQt5 import QtCore, QtGui, QtWidgets
from v4 import *
import os
import random

import mentalSQL

f = open('guiData/DataBaseFilePath.txt', 'r')
DataBaseFile = str(f.readline()) + 'mentalDB'


class Example(QMainWindow):

    def __init__(self):
        super().__init__()



        self.setWindowTitle('精神健康助手')
        self.resize(2000, 1200)
        self.setWindowIcon(QIcon('guiData/s.png'))
        palette = QtGui.QPalette()
        pix = QtGui.QPixmap("guiData/preview.jpg")
        pix = pix.scaled(self.width(), self.height())
        palette.setBrush(QtGui.QPalette.Background, QtGui.QBrush(pix))
        self.setPalette(palette)

        self.pageNum = 0
        self.center()
        self.initDBFileWay()
        self.initUI()
        self.show()

    def initUI(self):
        self.stacked = QStackedLayout(self)

        p0 = Page0()
        self.setCentralWidget(p0)
        p0.btn.clicked.connect(self.trun2_1)

    def trun2_1(self):
        p1 = Page1()
        self.setCentralWidget(p1)
        p1.btn.clicked.connect(self.trun2_0)
        self.pageNum = 1

    def trun2_0(self):
        p0 = Page0()
        self.setCentralWidget(p0)
        p0.btn.clicked.connect(self.trun2_1)
        self.pageNum = 0

    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def initDBFileWay(self):
        self.textEdit = QTextEdit()
        self.statusBar()

        openFile = QAction(QIcon('open.png'), '设置精神健康数据储存路径', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('设置精神健康数据所在的文件夹')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&设置路径')
        fileMenu.addAction(openFile)

    def showDialog(self):
        fname = QFileDialog.getExistingDirectory(self, '选择路径', './')
        if fname:
            f = open('guiData/DataBaseFilePath.txt', 'w')
            f.write(fname + '/')

class Page0(QWidget):
    def __init__(self):
        super(Page0, self).__init__()
        self.initUI()

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头

        self.slot_init()  # 初始化槽函数

        # 定义常数
        # 眼睛长宽比阈值

        self.EYE_AR_THRESH = 0.2
        # 打哈欠长宽比阈值
        self.MAR_THRESH = 0.5
        # 哈欠个数阈值
        self.MOUTH_FRAMES = 3
        # 初始化帧计数器和眨眼总数
        self.COUNTER = 0
        self.TOTAL = 0
        # 初始化帧计数器和打哈欠总数
        self.mCOUNTER = 0
        self.mTOTAL = 0
        # 帧数
        self.counter_x = 0
        self.counter_eye = 0

        detection_model_path = 'trainedModels/haarcascade_frontalface_default.xml'
        emotion_model_path = 'trainedModels/mini_XCEPTION.hdf5'

        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]

        # 初始化Dlib的人脸检测器（HOG），然后创建面部标志物预测
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r'trainedModels/shape_predictor_68_face_landmarks.dat')
        # 计算眼部精神程度（张开比例）

        # 分别获取左右眼面部标志的索引
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    '''程序界面布局'''

    def initUI(self):



        pix = QPixmap('scan.gif')

        lb1 = QLabel(self)
        lb1.setGeometry(200, 150, 961, 721)
        lb1.setPixmap(pix)
        lb1.setScaledContents(True)  # 自适应QLabel大小
        # pix = QPixmap('sexy.jpg')
        #
        # lb2 = QLabel(self)
        # lb2.setGeometry(0, 250, 500, 210)
        # lb2.setPixmap(pix)
        # lb2.setScaledContents(True)  # 自适应QLabel大小

        # 这种静态的方法设置一个用于显示工具提示的字体。我们使用10px滑体字体。
        QToolTip.setFont(QFont('SansSerif', 10))

        # 创建一个PushButton并为他设置一个tooltip
        self.btn = QPushButton('切换', self)
        self.btn.setToolTip('点击按钮将切换为<b>精神状态信息图表与相关建议</b>')

        # btn.sizeHint()显示默认尺寸
        self.btn.resize(self.btn.sizeHint())
        self.btn.move(1650, 950)

        self.button_open_camera = QPushButton('打开摄像头', self)  # 建立用于打开摄像头的按键
        self.button_open_camera.setMinimumHeight(80)
        self.button_open_camera.move(200, 950)
        self.button_open_camera.setFixedSize(250, 100)
        self.button_open_camera.setStyleSheet("QPushButton{\n"
                                    "    background:#008B8B;\n"
                                    "    color:white;\n"
                                    "    box-shadow: 1px 1px 3px;font-size:24px;border-radius: 24px;font-family: 微软雅黑;\n"
                                    "}\n"
                                    "QPushButton:pressed{\n"
                                    "    background:black;\n"
                                    "}")
        '''信息显示'''
        self.label_show_camera = QLabel(self)  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(961, 721)  # (641, 481)  # 给显示视频的Label设置大小为641x481
        # self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.move(200, 150)
        '''把按键加入到按键布局中'''

        self.bars = []
        x = 0
        for i in ('疲倦', '愤怒', '伤心', '中立', '开心', '惊奇', '厌恶', '恐惧'):
            l = QLabel(self)
            l.setText(i + '：')
            l.setFont(QFont('STZhongsong', 12))
            l.move(1300, 250 + x)

            self.bars.append(QProgressBar(self))
            self.bars[-1].setFixedSize(300, 30)
            self.bars[-1].move(1400, x + 260)

            x += 70

    def paintEvent(self, e) -> None:
        c1, c2 = QColor(68, 144, 196), QColor(165, 165, 165)
        qp = QPainter()
        qp.begin(self)

        qp.fillRect(200, 40, 800, 80, QBrush(c1, Qt.SolidPattern))

        qp.fillRect(1000, 40, 800, 80, QBrush(c2, Qt.SolidPattern))

        qp.setFont(QFont('STZhongsong', 18))
        qp.setPen(Qt.white)
        qp.drawText(400, 95, "实时精神健康数据")
        qp.drawText(1100, 95, "历史记录")

        qp.end()

    def slot_init(self):
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()

    '''槽函数之一'''

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭摄像头')

        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('打开摄像头')
            for i in self.bars:
                i.setValue(0)

    def show_camera(self):
        # flag, self.image = self.cap.read()  # 从视频流中读取

        # show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        # show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色

        haveFace = True
        td = 0

        ret, frame = self.cap.read()
        self.image = frame
        # reading the frame
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            mental_data = tuple(map(lambda x: int(x * 100), preds))

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
                # emoji_face = feelings_faces[np.argmax(preds)]

                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
        else:
            haveFace = False

        show = cv2.resize(frameClone, (960, 720))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色

        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

        # 进行循环，读取图片，并对图片做维度扩大，并进灰度化
        frame = imutils.resize(frame, width=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用detector(gray, 0) 进行脸部位置检测
        rects = self.detector(gray, 0)

        # 循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息

        self.counter_x += 1
        for rect in rects:

            shape = self.predictor(gray, rect)
            # 将脸部特征信息转换为数组array的格式
            shape = face_utils.shape_to_np(shape)
            # 提取左眼和右眼坐标
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            # 嘴巴坐标
            mouth = shape[self.mStart:self.mEnd]

            # 构造函数计算左右眼的OPEN值，使用平均值作为最终的OPEN
            leftEAR = eye_open_percent(leftEye)
            rightEAR = eye_open_percent(rightEye)
            open_e = (leftEAR + rightEAR) / 2.0
            # 打哈欠
            mar = mouth_open_percent(mouth)

            if open_e < EYE_AR_THRESH:
                self.counter_eye += 1

                # 计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
                # 同理，判断是否打哈欠
            if mar > MAR_THRESH:  # 张嘴阈值0.5
                self.mCOUNTER += 1
            else:
                # 如果连续3次都小于阈值，则表示打了一次哈欠
                if self.mCOUNTER >= MOUTH_FRAMES:  # 阈值：3
                    self.mTOTAL += 1
                    # 重置嘴帧计数器
                self.mCOUNTER = 0

            tired_degree = 90 * self.counter_eye / self.counter_x + 30 * (1 - 1 / (1 + self.mTOTAL))
            td = int(tired_degree)

        if haveFace:
            md = [mental_data[0], mental_data[4], mental_data[6], mental_data[3], mental_data[5], mental_data[1],
                  mental_data[2]]

            mentalSQL.setNewDatabaseOrSheet(DataBaseFile)
            mentalSQL.updateData(DataBaseFile, td, md)
            self.bars[0].setValue(td)
            for i in range(len(md)):
                self.bars[i + 1].setValue(md[i])


class Page1(QWidget):
    def __init__(self):
        super(Page1, self).__init__()
        self.initUI()

    def initUI(self):
        self.flag=False
        self.picmum=0

        # 这种静态的方法设置一个用于显示工具提示的字体。我们使用10px滑体字体。
        QToolTip.setFont(QFont('SansSerif', 10))

        # 创建一个PushButton并为他设置一个tooltip
        self.btn = QPushButton('切换', self)
        self.btn.setToolTip('点击按钮将切换为<b>实时精神健康数据</b>')

        # btn.sizeHint()显示默认尺寸
        self.btn.resize(self.btn.sizeHint())
        self.btn.move(1650, 1000)

        self.lbl = QLabel(self)
        self.lbl.move(200, 150)
        self.lbl.setText('尚未选择日期')

        cal = QCalendarWidget(self)
        cal.setGridVisible(True)
        cal.move(1350, 250)
        cal.clicked[QDate].connect(self.showDate)
        self.date = cal.selectedDate()

        self.dl = QLabel(self)
        self.dl.setFont(QFont('STZhongsong', 10))
        self.dl.setText('请在下表中选择您要查询的日期')
        self.dl.move(1350, 200)

        botton = QPushButton('确认查询', self)
        botton.move(1350, 650)
        botton.clicked.connect(self.search)

        botton_r = QPushButton('>', self)
        botton_r.move(800, 1075)
        botton_r.clicked.connect(self.change_pic_r)

        botton_l = QPushButton('<', self)
        botton_l.move(650, 1075)
        botton_l.clicked.connect(self.change_pic_l)

        b = QPushButton('查看小贴士', self)
        b.move(1350, 750)
        b.clicked.connect(self.xts)

    def showDate(self, date):
        self.date = date
        self.dl.setStyleSheet("color:black")
        self.dl.setText(f'您选择了{self.date.year()}年{self.date.month()}月{self.date.day()}日')

    def search(self):
        self.picmum=0
        conn = mentalSQL.sqlite3.connect(DataBaseFile + '.db')  # 数据库地址
        sql = f'SELECT * FROM Y{self.date.year()}M{self.date.month()}D{self.date.day()}'  # 表地址
        sur = conn.cursor()
        try:
            sur.execute(sql)
        except:
            self.dl.setStyleSheet("color:red")
            self.dl.setText(f'您选择的日期没有数据')

            return

        u = sur.fetchall()
        mentalSQL.drawTheMentalData(f'{self.date.year()}年{self.date.month()}月{self.date.day()}日', u)

        s = 1
        ls = os.listdir("guiData/bars")
        pic_list=[]
        for i in ls:
            c_path = os.path.join("guiData/bars", i)
            pic_list.append(c_path)
        print(self.picmum)
        try:
            pixmap = QPixmap(pic_list[self.picmum])
            self.lbl.setPixmap(pixmap)  # 在label上显示图片
            self.lbl.resize(900 * s, 897 * s)
            self.lbl.setScaledContents(True)  # 让图片自适应label大小
            self.flag = True
        except:
            self.dl.setStyleSheet("color:red")
            self.dl.setText(f'您选择的日期没有数据')# 按指定路径找到图片


    def change_pic_r(self):
        if self.flag==False:
            QtWidgets.QMessageBox.information(self, 'Ooops!', "您还未选择日期")
        else:
            s = 1
            ls = os.listdir("guiData/bars")
            pic_list = []
            for i in ls:
                c_path = os.path.join("guiData/bars", i)
                pic_list.append(c_path)
            self.picmum+=1
            if self.picmum<len(pic_list):
                pixmap = QPixmap(pic_list[self.picmum])  # 按指定路径找到图片
                self.lbl.setPixmap(pixmap)  # 在label上显示图片
                self.lbl.resize(900 * s, 897 * s)
                self.lbl.setScaledContents(True)  # 让图片自适应label大小
            else:
                self.picmum-=1
                QtWidgets.QMessageBox.information(self, 'Ooops!', "已经是最后一张啦")
    def change_pic_l(self):
        if self.flag == False:
            QtWidgets.QMessageBox.information(self, 'Ooops!', "您还未选择日期")
        else:
            s = 1
            ls = os.listdir("guiData/bars")
            pic_list = []
            for i in ls:
                c_path = os.path.join("guiData/bars", i)
                pic_list.append(c_path)
            self.picmum -= 1
            if self.picmum >= 0:
                pixmap = QPixmap(pic_list[self.picmum])  # 按指定路径找到图片
                self.lbl.setPixmap(pixmap)  # 在label上显示图片
                self.lbl.resize(900 * s, 897 * s)
                self.lbl.setScaledContents(True)  # 让图片自适应label大小
            else:
                self.picmum += 1
                QtWidgets.QMessageBox.information(self, 'Ooops!', "已经是第一张啦")


    def paintEvent(self, e) -> None:
        c2, c1 = QColor(68, 144, 196), QColor(165, 165, 165)
        qp = QPainter()
        qp.begin(self)

        qp.fillRect(200, 40, 800, 80, QBrush(c1, Qt.SolidPattern))

        qp.fillRect(1000, 40, 800, 80, QBrush(c2, Qt.SolidPattern))

        qp.setFont(QFont('STZhongsong', 18))
        qp.setPen(Qt.white)
        qp.drawText(400, 95, "实时精神健康数据")
        qp.drawText(1100, 95, "历史记录")

        qp.end()

    def xts(self):
        f = open('guiData/xts.txt', 'r', encoding='utf-8')
        x = tuple(f.readlines())
        s = x[random.randint(0, len(x) - 1)]

        QtWidgets.QMessageBox.information(self, '小贴士', s)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
