# Deep Face 

### Deep Face란
- Deepface는 Python용 경량 얼굴 인식 및 얼굴 속성 분석(나이,성별, 감정 및 인종) 프레임워크 이다.
- 최신 모델 VGG-Face인, google FaceNet, Openface, Facebook DeepFace, DeepID,및 ArcFace을 래핑하는 하이브리드 얼굴 인식 프레임워크이다.
- 이 라이브러리는 FaceNet 및 InsightFace와 같은 다양한 얼굴 인식 방법을 지원하며 REST API도 제공하지만 인증 방식만 지원하기 때문에 얼굴 모음을 생성하고 그중에서 얼굴을 찾을 수 없다
- 파이썬 개발자라면 쉽게 시작할 수 있지만 다른 사람들이 통합하기는 어려울 수 있다.

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
model_name = models[1]
#-- Facenet 선택
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
def stream(...):
    if time_threshold < 0:
        raise ValueError("time_threshold must be greater than the value 1 but you passed" + str(time_threshold))
    if frame_threshold < 0;
        raise ValueError("frame_threshold must be greater than the value 1 but you passed" + str(time_threshold))
```

### 사용법_3 - realtime.py

- def stream()함수에서 realtime.analysis() 부분을 ctrl+클릭을 눌러서 def analysis()함수 수정

```python
def analysis(...):

	face_detector = FaceDetector.build_model(detector_backend)
	print("Detector backend is ", detector_backend)
	#------------------------
	input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

    #------------------------------------------- 수정 -----------------------------------------------------------#
	faceIdColor = (0,255,0) #-- 초록
    unknowColor = (0,0,255) #-- 빨강
    #------------------------------------------- 수정 -----------------------------------------------------------#

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
    
    
    #-- 학습할 이미지가 없다면 오류 발생
	if len(employees) == 0:
		print("WARNING: There is no image in this path ( ", db_path,") . Face recognition will not be performed.")


    #-- 학습할 이미지가 있다면
	if len(employees) > 0:

		model = DeepFace.build_model(model_name)
		print(model_name," is built")

		#------------------------

		input_shape = functions.find_input_shape(model)
		input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

		#tuned thresholds for model and metric pair
		threshold = dst.findThreshold(model_name, distance_metric)

	#------------------------
	#facial attribute analysis models

    #------------------------------------------- 감정 분석 등이 필요하면 주석 해제 ------------------------------------------------#
	# if enable_face_analysis == True:

	# 	tic = time.time()

	# 	emotion_model = DeepFace.build_model('Emotion')
	# 	print("Emotion model loaded")

	# 	age_model = DeepFace.build_model('Age')
	# 	print("Age model loaded")

	# 	gender_model = DeepFace.build_model('Gender')
	# 	print("Gender model loaded")

	# 	toc = time.time()

	# 	print("Facial attribute analysis models loaded in ",toc-tic," seconds")
    #------------------------------------------- 감정 분석 등이 필요하면 주석 해제 ------------------------------------------------#



	tic = time.time()
	
    #------ 이미지 학습 후 임베딩 진행 

    pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')
	#TODO: why don't you store those embeddings in a pickle file similar to find function?

	embeddings = []
	#for employee in employees:
	for index in pbar:
		employee = employees[index]
		pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
		embedding = []

		#preprocess_face returns single face. this is expected for source images in db.
		img = functions.preprocess_face(img = employee, target_size = (input_shape_y, input_shape_x), enforce_detection = False, detector_backend = detector_backend)
		img_representation = model.predict(img)[0,:]

		embedding.append(employee)
		embedding.append(img_representation)
		embeddings.append(embedding)

	df = pd.DataFrame(embeddings, columns = ['employee', 'embedding'])
	df['distance_metric'] = distance_metric

	toc = time.time()

	print("Embeddings found for given data set in ", toc-tic," seconds")



	pivot_img_size = 112 # 인식된 결과 이미지 크기 

	
	freeze = False
	face_detected = False
	face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
	freezed_frame = 0
	tic = time.time()

	cap = cv2.VideoCapture(source) #웹캠은 0으로 설정

    
	while(True):
		ret, img = cap.read()

		if img is None:
			break

		#cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
		#cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

		raw_img = img.copy()
		resolution = img.shape; resolution_x = img.shape[1]; resolution_y = img.shape[0]

		if freeze == False:
			try:
				#faces store list of detected_face and region pair
				faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = False)
			except: #to avoid exception if no face detected
				faces = []

			if len(faces) == 0:
				face_included_frames = 0
		else:
			faces = []

		detected_faces = []
		face_index = 0
        
        #-- 얼굴이 인식된경우 --#
		for face, (x, y, w, h), _ in faces:
			if w > 130: #작은 이미지는 버림 

				face_detected = True
				if face_index == 0:
					face_included_frames = face_included_frames + 1 #increase frame for a single face

				cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

				cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

				detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

				#-------------------------------------

				detected_faces.append((x,y,w,h))
				face_index = face_index + 1

				#-------------------------------------

		if face_detected == True and freeze == False:
			freeze = True
			#base_img = img.copy()
			base_img = raw_img.copy()
			detected_faces_final = detected_faces.copy()
			tic = time.time()

        #-- 얼굴이 인식된 경우 --#
		if freeze == True:

			toc = time.time()
			if (toc - tic) < time_threshold:

				if freezed_frame == 0:
					freeze_img = base_img.copy()
					#freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue

					for detected_face in detected_faces_final:
						x = detected_face[0]; y = detected_face[1]
						w = detected_face[2]; h = detected_face[3]

						cv2.rectangle(freeze_img, (x,y), (x+w,y+h), (67,67,67), 1) #인식된 얼굴 사각박스

						custom_face = base_img[y:y+h, x:x+w]


                        #---------------------------------- 감정 분석 등이 필요하면 주석 해제 -------------------------------------------#
						# if enable_face_analysis == True:

						# 	gray_img = functions.preprocess_face(img = custom_face, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'opencv')
						# 	emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
						# 	emotion_predictions = emotion_model.predict(gray_img)[0,:]
						# 	sum_of_predictions = emotion_predictions.sum()

						# 	mood_items = []
						# 	for i in range(0, len(emotion_labels)):
						# 		mood_item = []
						# 		emotion_label = emotion_labels[i]
						# 		emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
						# 		mood_item.append(emotion_label)
						# 		mood_item.append(emotion_prediction)
						# 		mood_items.append(mood_item)

						# 	emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
						# 	emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)

						# 	#background of mood box

						# 	#transparency
						# 	overlay = freeze_img.copy()
						# 	opacity = 0.4

						# 	if x+w+pivot_img_size < resolution_x:
						# 		#right
						# 		cv2.rectangle(freeze_img
						# 			#, (x+w,y+20)
						# 			, (x+w,y)
						# 			, (x+w+pivot_img_size, y+h)
						# 			, (64,64,64),cv2.FILLED)

						# 		cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

						# 	elif x-pivot_img_size > 0:
						# 		#left
						# 		cv2.rectangle(freeze_img
						# 			#, (x-pivot_img_size,y+20)
						# 			, (x-pivot_img_size,y)
						# 			, (x, y+h)
						# 			, (64,64,64),cv2.FILLED)

						# 		cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

						# 	for index, instance in emotion_df.iterrows():
						# 		emotion_label = "%s " % (instance['emotion'])
						# 		emotion_score = instance['score']/100

						# 		bar_x = 35 #this is the size if an emotion is 100%
						# 		bar_x = int(bar_x * emotion_score)

						# 		if x+w+pivot_img_size < resolution_x:

						# 			text_location_y = y + 20 + (index+1) * 20
						# 			text_location_x = x+w

						# 			if text_location_y < y + h:
						# 				cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

						# 				cv2.rectangle(freeze_img
						# 					, (x+w+70, y + 13 + (index+1) * 20)
						# 					, (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)
						# 					, (255,255,255), cv2.FILLED)

						# 		elif x-pivot_img_size > 0:

						# 			text_location_y = y + 20 + (index+1) * 20
						# 			text_location_x = x-pivot_img_size

						# 			if text_location_y <= y+h:
						# 				cv2.putText(freeze_img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

						# 				cv2.rectangle(freeze_img
						# 					, (x-pivot_img_size+70, y + 13 + (index+1) * 20)
						# 					, (x-pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)
						# 					, (255,255,255), cv2.FILLED)

						# 	#-------------------------------

						# 	face_224 = functions.preprocess_face(img = custom_face, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv')

						# 	age_predictions = age_model.predict(face_224)[0,:]
						# 	apparent_age = Age.findApparentAge(age_predictions)

						# 	#-------------------------------

						# 	gender_prediction = gender_model.predict(face_224)[0,:]

						# 	if np.argmax(gender_prediction) == 0:
						# 		gender = "W"
						# 	elif np.argmax(gender_prediction) == 1:
						# 		gender = "M"

						# 	#print(str(int(apparent_age))," years old ", dominant_emotion, " ", gender)

						# 	analysis_report = str(int(apparent_age))+" "+gender

						# 	#-------------------------------

						# 	info_box_color = (46,200,255)

						# 	#top
						# 	if y - pivot_img_size + int(pivot_img_size/5) > 0:

						# 		triangle_coordinates = np.array( [
						# 			(x+int(w/2), y)
						# 			, (x+int(w/2)-int(w/10), y-int(pivot_img_size/3))
						# 			, (x+int(w/2)+int(w/10), y-int(pivot_img_size/3))
						# 		] )

						# 		cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

						# 		cv2.rectangle(freeze_img, (x+int(w/5), y-pivot_img_size+int(pivot_img_size/5)), (x+w-int(w/5), y-int(pivot_img_size/3)), info_box_color, cv2.FILLED)

						# 		cv2.putText(freeze_img, analysis_report, (x+int(w/3.5), y - int(pivot_img_size/2.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

						# 	#bottom
						# 	elif y + h + pivot_img_size - int(pivot_img_size/5) < resolution_y:

						# 		triangle_coordinates = np.array( [
						# 			(x+int(w/2), y+h)
						# 			, (x+int(w/2)-int(w/10), y+h+int(pivot_img_size/3))
						# 			, (x+int(w/2)+int(w/10), y+h+int(pivot_img_size/3))
						# 		] )

						# 		cv2.drawContours(freeze_img, [triangle_coordinates], 0, info_box_color, -1)

						# 		cv2.rectangle(freeze_img, (x+int(w/5), y + h + int(pivot_img_size/3)), (x+w-int(w/5), y+h+pivot_img_size-int(pivot_img_size/5)), info_box_color, cv2.FILLED)

						# 		cv2.putText(freeze_img, analysis_report, (x+int(w/3.5), y + h + int(pivot_img_size/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

						#---------------------------------- 감정 분석 등이 필요하면 주석 해제 -------------------------------------------#
						
                        
                        #---- 얼굴 인식  ----#

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
                                
                                display_img = cv2.imread(employee_name) #-- 보여줄 이미지 선택
								display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
                                
								if best_distance <= threshold: #-- 학습된 인물 --#
									#print(employee_name)

									label = employee_name.split("/")[-1].replace(".jpg", "")
									label = re.sub('[0-9]', '', label)

									try: #-- 인식된 객체의 위치에 따라 텍스트위치 수정
										if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
											#우측 상단
											freeze_img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img

											cv2.putText(freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, faceIdColor, 1)

											#사진과 라인 연결 
											cv2.line(freeze_img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
											cv2.line(freeze_img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)

										elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
											#왼쪽 하단
											freeze_img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img

											cv2.putText(freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, faceIdColor, 1)

											#사진과 라인 연결 
											cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
											cv2.line(freeze_img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),1)

										elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
											#좌측 상단
											freeze_img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img

											cv2.putText(freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, faceIdColor, 1)

											#사진과 라인 연결
											cv2.line(freeze_img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
											cv2.line(freeze_img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),1)

										elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
											#우측 하단
											freeze_img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img

											cv2.putText(freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, faceIdColor, 1)

											#사진과 라인 연결
											cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
											cv2.line(freeze_img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),1)
									except Exception as err:
										print(str(err))
                                
                                else: #-- 학습되지 않은 인물 --#
                                    try: #-- 인식된 객체의 위치에 따라 텍스트위치 수정
										if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
											#우측 상단
											freeze_img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img

											cv2.putText(freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, unknowColor, 1)

											#사진과 라인 연결 
											cv2.line(freeze_img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
											cv2.line(freeze_img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)

										elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
											#왼쪽 하단
											freeze_img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img

											cv2.putText(freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, unknowColor, 1)

											#사진과 라인 연결 
											cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
											cv2.line(freeze_img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),1)

										elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
											#좌측 상단
											freeze_img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img

											cv2.putText(freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, unknowColor, 1)

											#사진과 라인 연결
											cv2.line(freeze_img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
											cv2.line(freeze_img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),1)

										elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
											#우측 하단
											freeze_img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img

											cv2.putText(freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, unknowColor, 1)

											#사진과 라인 연결
											cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
											cv2.line(freeze_img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),1)
									except Exception as err:
										print(str(err))

						tic = time.time() #in this way, freezed image can show 5 seconds

				time_left = int(time_threshold - (toc - tic) + 1)

				cv2.rectangle(freeze_img, (10, 10), (90, 50), (67,67,67), -10)
				cv2.putText(freeze_img, "Face ID ", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

				cv2.imshow('img', freeze_img)

				freezed_frame = freezed_frame + 1
			else:
				face_detected = False
				face_included_frames = 0
				freeze = False
				freezed_frame = 0

		else:
            cv2.putText(freeze_img, "Face ID ", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
			cv2.imshow('img',img)

		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break

	#kill open cv things
	cap.release()
	cv2.destroyAllWindows()

```