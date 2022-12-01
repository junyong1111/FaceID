# FaceID

1. DeepFace
Facial a.
6. Facial a.
b.
c.
[개발 피드백]
Recognition 의 한계
대부분 데이터셋이 서양인에 맞추어져 있어 동양인에 대한 얼굴 인식률이 낮다.
Recognition 기능 개선을 위한 방법론 탐구 Ageitgey/face_recognition
i. 현재 사용 중인 모델로 빠르고 가볍지만 동양인과 흑인 인식률이 상당히 낮음 CompreFace
i. FaceNet과 InsightFace를 사용하며 docker를 사용
ii. 정보가 많이 없어서 사용하기 어려움
iii. 도커를 사용하여 실행하였으나 Python 제어가 힘듬
DeepFace
i. Deepface는 Python용 경량 얼굴 인식 및 얼굴 속성 분석(나이, 성별, 이다.
감정 및 인종) 프레임워크
ii. 최신 모델 VGG-Face인, google FaceNet, Openface, Facebook DeepFace, DeepID,및 ArcFace을 래핑하는 하이브리드 얼굴 인식 프레임워크이다.
iii. 이 라이브러리는 FaceNet 및 InsightFace와 같은 다양한 얼굴 인식 방법을 지원하며 REST API도 제공하지만 인증 방식만 지원하기 때문에 얼굴 모음을 생성하고 그중에서 얼굴을 찾을 수 없다
iv. 파이썬 개발자라면 쉽게 시작할 수 있지만 다른 사람들이 통합하기는 어려울 수 있다.
이슈 : 현재 사용 중인 FaceRecognition 모델은 2018년까지가 최신 업데이트 모델로 서양인을 제외한 나머지 인종에 대한
얼굴인식률이 상당히 낮다. 따라서 모델에 대한 교체 필요성을 느꼈고 다음 5가지 얼굴인식 모델을 후보로 생각
1. CompreFace
2. DeepFace
3. FaceNet
4. InsightFace
5. InsightFace-REST
해결 : 3~5번까지 얼굴인식 모델이 포함되어 있는 프레임워크인 CompreFace 또는 DeepFace를 선택 둘 중 현재 정보나
