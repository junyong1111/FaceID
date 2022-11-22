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
    
```