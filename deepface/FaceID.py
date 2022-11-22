from deepface import DeepFace

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "ArcFace",
    "Dlib",
    "SFace",
]
model_name = models[1]

metricsList = ["cosine", "euclidean","euclidean_12"]
metrics = metricsList[2]

