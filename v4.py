from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance


# parameters for loading data and images
detection_model_path = 'trainedModels/haarcascade_frontalface_default.xml'
emotion_model_path = 'trainedModels/mini_XCEPTION.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

injshu = 1

# 定义常数
# 眼睛长宽比阈值
EYE_AR_THRESH = 0.2
# 打哈欠长宽比阈值
MAR_THRESH = 0.5
# 哈欠个数阈值
MOUTH_FRAMES = 3
# 初始化帧计数器和眨眼总数
COUNTER = 0
TOTAL = 0
# 初始化帧计数器和打哈欠总数
mCOUNTER = 0
mTOTAL = 0
# 帧数
counter_x = 0
counter_eye = 0


# 初始化Dlib的人脸检测器（HOG），然后创建面部标志物预测
# print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'trainedModels/shape_predictor_68_face_landmarks.dat')
# 计算眼部精神程度（张开比例）
def eye_open_percent(eye):
    e1 = distance.euclidean(eye[1], eye[5])
    e2 = distance.euclidean(eye[2], eye[4])
    # 计算水平之间的距离
    dist_eye = distance.euclidean(eye[0], eye[3])
    e_open = (e1 + e2) / (2.0 * dist_eye)
    return e_open


# 计算嘴部张开比例
def mouth_open_percent(mouth):  # 嘴部
    m1 = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    m2 = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    m3 = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    m = (m1 + m2) / (2.0 * m3)
    return m


# 分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

'''# 打开cv2 本地摄像头
# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)


td = 0
mental_data = []

while True:

    frame = camera.read()[1]
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

        mental_data = tuple(map(lambda x:int(x*100),preds))
    else:
        continue

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
    #    for c in range(0, 3):
    #        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
    #        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
    #        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)

    cv2.imshow('your_face', frameClone)
    # cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    counter_x += 1
    temp_k_flag = 0
    # 进行循环，读取图片，并对图片做维度扩大，并进灰度化
    ret, frame = camera.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用detector(gray, 0) 进行脸部位置检测
    rects = detector(gray, 0)

    # 循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        shape = predictor(gray, rect)
        # 将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)
        # 提取左眼和右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # 嘴巴坐标
        mouth = shape[mStart:mEnd]

        # 构造函数计算左右眼的OPEN值，使用平均值作为最终的OPEN
        leftEAR = eye_open_percent(leftEye)
        rightEAR = eye_open_percent(rightEye)
        open_e = (leftEAR + rightEAR) / 2.0
        # 打哈欠
        mar = mouth_open_percent(mouth)

        # 进行画图操作，用矩形框标注人脸
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

        if open_e < EYE_AR_THRESH:
            counter_eye += 1

        # 进行画图操作，同时使用cv2.putText将眨眼次数进行显示
        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "eye degree: {:.2f}".format(100*counter_eye/counter_x), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),2)

        # 计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
        # 同理，判断是否打哈欠
        if mar > MAR_THRESH:  # 张嘴阈值0.5
            mCOUNTER += 1
        else:
            # 如果连续3次都小于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_FRAMES:  # 阈值：3
                mTOTAL += 1
            # 重置嘴帧计数器
            mCOUNTER = 0

        cv2.putText(frame, "Yawning: {}".format(mTOTAL), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        tired_degree = 90*counter_eye/counter_x+30*(1-1/(1+mTOTAL))
        td = int(tired_degree)
        cv2.putText(frame, "Tired degree: {:.2f}".format(tired_degree), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    mentalSQL.setNewDatabaseOrSheet(DataBaseFile)
    mentalSQL.updateData(DataBaseFile, td, mental_data)

    # 按q退出
    # cv2.putText(frame, "Press 'q': Quit", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # 窗口显示 show with opencv
    # cv2.imshow("Frame", frame)

    # if the `q` key was pressed, break from the loop
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

camera.release()
cv2.destroyAllWindows()
'''