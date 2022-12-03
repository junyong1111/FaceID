# Deep Face 

### Deep Face란
- Deepface는 Python용 경량 얼굴 인식 및 얼굴 속성 분석(나이,성별, 감정 및 인종) 프레임워크 이다.
- 최신 모델 VGG-Face인, google FaceNet, Openface, Facebook DeepFace, DeepID,및 ArcFace을 래핑하는 하이브리드 얼굴 인식 프레임워크이다.
- 이 라이브러리는 FaceNet 및 InsightFace와 같은 다양한 얼굴 인식 방법을 지원하며 REST API도 제공하지만 인증 방식만 지원하기 때문에 얼굴 모음을 생성하고 그중에서 얼굴을 찾을 수 없다
- 파이썬 개발자라면 쉽게 시작할 수 있지만 다른 사람들이 통합하기는 어려울 수 있다.


#### 기존 face_recognitio은 동양인은 인식이 잘 되지않는다는 문제점이 존재했지만 해당 모델은 동양인 얼굴에서도 높은 인식률을 보여줬다.
#### 모델은 FaceNet이 인식률이 제일 좋았고 얼굴 탐지는 opencv가 정확하지는 않지만 빨라서 쓰기 좋았다.


### 파일구조

```bash
-FaceID.py
-SaveimgFromVideo.py
-README.md
-UserName
⎿  UserName1.jpg
⎿  UserName2.jpg
⎿  UserName3.jpg
⎿  ...
```

### 학습 이미지 준비 - SaveimgFromVideo.py

- def SaveImg(path, name , maximumPic, fps)를 사용하여 저장된 동영상을 이미지로 분할하여 저장
```python
#-- 동영상 경로/유저 이름/최대 저장 수/몇 프레임마다 저장할지
def SaveImg(path, name , maximumPic, fps)
```


### 설치 

- 다음 명령어를 통해 쉽게 설치 가능
```bash
pip install deepface
```

### 사용법_1 - FaceID.py

1. 필요 라이브러리 import 
```python
from deepface import DeepFace
```

2. 해당 모델 중 원하는 모델을 선택
```python
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "ArcFace",
    "Dlib",
    'SFace",
]
model_name = models[8]
#-- SFace 선택
```

3. 유사성 선택

```python
metricsList = ["cosine", "euclidean","euclidean_12"]

metrics = metricsList[2]
#-- euclidean_12 선택
```
Euclidean L2 형태 는 실험을 기반으로 cosine 및 regular Euclidean distance보다 더 안정적

4. 실시간 얼굴 인식

```python
DeepFace.stream(db_path = "학습할 얼굴 이미지 폴더", distance_metric = metrics, time_threshold = 0, frame_threshold = 0,
                model_name = model_name, enable_face_analysis = False)
```
- enable_face_analysis = True로 사용하면 감정, 나이, 성별 등 다양한 분석 가능 얼굴인식만 필요하므로 False 설정



### 사용법_2 - DeepFace.py

- DeepFace.stream 함수에서 stream 부분을 ctrl+클릭을 눌러서 def stream()함수 수정

```python
#-- def stream()함수 수정
def stream(model, threshold , df,  db_path = '', model_name ='VGG-Face', detector_backend = 'opencv', distance_metric = 'cosine', enable_face_analysis = True, source = 0, time_threshold = 5, frame_threshold = 5):
    if time_threshold < 0:
        raise ValueError("time_threshold must be greater than the value 1 but you passed" + str(time_threshold))
    if frame_threshold < 0;
        raise ValueError("frame_threshold must be greater than the value 1 but you passed" + str(time_threshold))
	return realtime.analysis(dp_path, model_name, detector_backend, distance_metric, enable_face_analysis, source = source, time_threshold =time_threshold, frame_threshold  	=frame_threshold)
```

### 사용법_3 - realtime.py

- def stream()함수에서 realtime.analysis() 부분을 ctrl+클릭을 눌러서 def analysis()함수 수정

```python
def analysis(model, threshold, df, db_path, model_name = 'VGG-Face', detector_backend = 'opencv', distance_metric = 'cosine', enable_face_analysis = True, source = 0, time_threshold = 5, frame_threshold = 5):
	label = "Stranger"
	#-----------------------
	strangerColor = (0,0,255)
 	
	face_detector = FaceDetector.build_model(detector_backend)
	pivot_img_size = 112 #face recognition result image

	#-----------------------
	userColor = (0,255,0)
 	
 
	face_detected = False
	
	input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
	# tic = time.time()
 
	img = source
	raw_img = img.copy()
 
	resolution = img.shape; resolution_x = img.shape[1]; resolution_y = img.shape[0]
	faces = []
	try:
		#faces store list of detected_face and region pair
		faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = False)
	except: #to avoid exception if no face detected
		faces = []
	

	detected_faces = []
	face_index = 0
	for face, (x, y, w, h) in faces:
		if w > 130: #discard small detected faces
			#-- 얼굴이 인식된 경우 
			face_detected = True
		 #increase frame for a single face

			cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2) #draw rectangle to main image
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			#-------------------------------------

			detected_faces.append((x,y,w,h))
			face_index = face_index + 1

			#-------------------------------------

	if face_detected == True:
		#base_img = img.copy()
		base_img = raw_img.copy()
		detected_faces_final = detected_faces.copy()
  
		# tic = time.time()
		# if (toc - tic) < time_threshold:
      
		freeze_img = base_img.copy()
		#freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue

		for detected_face in detected_faces_final:
			x = detected_face[0]; y = detected_face[1]
			w = detected_face[2]; h = detected_face[3]

			cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (0,0,255), 1) #draw rectangle to main image

			#-------------------------------

			#apply deep learning for custom_face

			custom_face = base_img[y:y+h, x:x+w]

			#-------------------------------
			#face recognition

			custom_face = functions.preprocess_face(img = custom_face, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = 'opencv')

			#check preprocess_face function handled
			if custom_face.shape[1:3] == input_shape:
				if df.shape[0] > 0: #if there are images to verify, apply face recognition
					img1_representation = model.predict(custom_face)[0,:]

					#print(freezed_frame," - ",img1_representation[0:5])

					def findDistance(row):
						distance_metric = row['distance_metric']
						img2_representation = row['embedding']

						distance = 1000 #initialize very large value
						if distance_metric == 'cosine':
							distance = dst.findCosineDistance(img1_representation, img2_representation)
						elif distance_metric == 'euclidean':
							distance = dst.findEuclideanDistance(img1_representation, img2_representation)
						elif distance_metric == 'euclidean_l2':
							distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))

						return distance

					df['distance'] = df.apply(findDistance, axis = 1)
					df = df.sort_values(by = ["distance"])

					candidate = df.iloc[0]
					employee_name = candidate['employee']
					best_distance = candidate['distance']

					#print(candidate[['employee', 'distance']].values)

					#if True:
					print("threshold is", threshold)
					print("bt distance is", best_distance)
					if best_distance <= threshold:
						#print(employee_name)
						display_img = cv2.imread(employee_name)
						display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))

						label = employee_name.split("/")[-1].replace(".jpg", "")
						label = re.sub('[0-9]', '', label)
      
						cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (255,0,0), 1)

						try:
							if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
								#top right
								# freeze_img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img
								cv2.putText(freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, userColor, 2)
							elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
								#bottom left
								# freeze_img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img
								cv2.putText(freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, userColor, 2)
							elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
								#top left
								# freeze_img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img
							
								cv2.putText(freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, userColor, 2)
							elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
								#bottom righ
								# freeze_img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img
								cv2.putText(freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, userColor, 2)
						except Exception as err:
							print(str(err))
					else :
						try:
							if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
								#top right
								# freeze_img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img
								cv2.putText(freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, strangerColor, 2)
							elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
								#bottom left
								# freeze_img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img
								cv2.putText(freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, strangerColor, 2)
							elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
								#top left
								# freeze_img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img
							
								cv2.putText(freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, strangerColor, 2)
							elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
								#bottom righ
								# freeze_img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img
								cv2.putText(freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, strangerColor, 2)
						except Exception as err:
							print(str(err))
				# tic = time.time() #in this way, freezed image can show 5 seconds

				#-------------------------------


			cv2.rectangle(freeze_img, (10, 10), (90, 50), (67,67,67), -10)
			cv2.imshow('img', freeze_img)

			face_detected = False
			return label
   
	
	else:
		cv2.imshow('img',img)
		return "Not Human"
```

### 사용법_3 - FaceEmbeddingfromImage.py
- 얼굴 학습 시 얼굴탐지 모델은 ssd로 고정(빠름)

```python
import os
import re
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from deepface import DeepFace
from deepface.extendedmodels import Age
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector

def FaceEmbedding(db_path, distance_metric, model_name, detector_backend):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    employees = []
    #check passed db folder exists
    if os.path.isdir(db_path) == True:
        for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file):
                    #exact_path = os.path.join(r, file)
                    exact_path = r + "/" + file
                    #print(exact_path)
                    employees.append(exact_path)

    if len(employees) == 0:
        print("WARNING: There is no image in this path ( ", db_path,") . Face recognition will not be performed.")
    
    if len(employees) > 0:
        model = DeepFace.build_model(model_name)
        print(model_name," is built")
        input_shape = functions.find_input_shape(model)
        input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
        threshold = dst.findThreshold(model_name, distance_metric)
        # tic = time.time()


    pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')
    #TODO: why don't you store those embeddings in a pickle file similar to find function?

    embeddings = []
    #for employee in employees:
    for index in pbar:
        employee = employees[index]
        pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
        embedding = []

        #preprocess_face returns single face. this is expected for source images in db.
        img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = 'opencv')
        img_representation = model.predict(img)[0,:]

        embedding.append(employee)
        embedding.append(img_representation)
        embeddings.append(embedding)

    df = pd.DataFrame(embeddings, columns = ['employee', 'embedding'])
    df['distance_metric'] = distance_metric

    # toc = time.time()

    print("Embeddings found for given data set")

    return threshold, df, model
#-----------------------


```