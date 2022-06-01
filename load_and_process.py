import pandas as pd
import cv2
import numpy as np


dataset_path = 'fer2013/fer2013/fer2013.csv'
#文件保存位置
image_size=(48,48)
#图片大小

def load_fer2013():
        data = pd.read_csv(dataset_path)
        #读取数据，pd中的函数，参数可以为文件路径，URL，临时文件
        pixels = data['pixels'].tolist()
        #numpy的函数，用于将矩阵转化为列表，将数据集中的pixel像素点部分，转换成列表，便于之后操作
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            #最终出一个大列表，一个一个矩阵，每个数据32位浮点型
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            #对每一幅图片的像素数组中，以空格断开
            face = np.asarray(face).reshape(width, height)
            #将face列表转化为宽高为48*48的矩阵，相当于还原了像素点的位置
            face = cv2.resize(face.astype('uint8'),image_size)
            #将face中的数据格式转化为8位无符号整型
            faces.append(face.astype('float32'))
            #将face以32位浮点型的格式填到faces中，那为什么要先转到8位无符号，在转到32位浮点？？直接把face转化为32位放到faces不行吗？

        faces = np.asarray(faces)
        #再将其转化为矩阵
        faces = np.expand_dims(faces, -1)
        #矩阵的shape表示矩阵每一维度的大小，比如【【【1,2】，【1，2】】】的shape为（1，2,3），有四个间隔，dims函数表示在这些间隔上插一维度，以刚刚的例子为例，参数为0，则为（1,1,2,3），参数为2，则为（1,2,1,3）为-1，贼在末尾插入。
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        #将数据集中情绪中的每一中情绪，作为数据库的属性。比如原本性别，内容为男、女。使用该函数后，则为性别——男、性别——女，内容为0、1.
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    #转化为浮点型
    x = x / 255.0
    #将像素转化为百分比，灰度图？
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x