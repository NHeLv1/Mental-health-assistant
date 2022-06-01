# -*- coding: utf-8 -*-
"""
运行本项目需要安装的库：
    keras 2.2.4
    PyQt5 5.11.3
    pandas 0.24.2
    scikit-learn 0.21.2
    tensorflow 1.13.1
    imutils 0.5.2
    opencv-python 4.10.25
    matplotlib 3.2.1  # 注意：此依赖包为第二版新增，请注意安装

点击运行主程序runMain.py
"""

import warnings
import os
from IPython.external.qt_for_kernel import QtCore
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')
from EmotionRecongnition import Ui_MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QCoreApplication


if __name__ == '__main__':
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(argv)

    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())
