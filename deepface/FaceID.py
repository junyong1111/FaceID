import cv2
import os
import tensorflow as tf
from deepface import DeepFace
from SaveimgFromVideo import SaveImg
from FaceEmbeddingFromImage import FaceEmbedding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

modelList = [
   "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]
backendList =[
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe'
]
metricList = ["cosine", "euclidean", "euclidean_l2"]


modelName = modelList[8] #-- SFace가 가장 잘 되는듯
backendName = backendList[4] #-- opnecv
metricName = metricList[2] #--euclidean_l2

userName = "CarOwner"

# print(userName)

SaveImg("user.mp4", userName, 15, 20)

threshold, df, model = FaceEmbedding(userName, distance_metric= metricName, model_name= modelName, detector_backend=backendName)

cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Not Frame")
        break
    ID = DeepFace.stream(model = model, threshold = threshold, df = df, db_path=userName, distance_metric= metricName,
                    time_threshold= 0.1, frame_threshold=0.1, model_name=modelName, enable_face_analysis=False, source=frame)
    print("ID is ", ID)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break