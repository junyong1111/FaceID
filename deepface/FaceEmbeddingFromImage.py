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
        tic = time.time()


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

    toc = time.time()

    print("Embeddings found for given data set in ", toc-tic," seconds")

    return threshold, df, model
#-----------------------

