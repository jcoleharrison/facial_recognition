from numpy import asarray
from numpy import expand_dims
import tensorflow as tf
import numpy as np
import pickle
import cv2
import re
import os
from architecture import * 
from mtcnn import MTCNN

os.chdir('src/v2/')
#Create detector model
detector = MTCNN()

# Create the FaceNet model
face_encoder = InceptionResNetV2()
face_encoder.load_weights('facenet_keras.h5')

# Load the dataset
with open("data.pkl", "rb") as myfile:
    database = pickle.load(myfile) 
    myfile.close()

# Live test
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    no_face = False
    _, image = cap.read()
    
    faces = detector.detect_faces(image)
    
    for i in range(len(faces)):
        if len(faces)>0:
            x1, y1, width, height = faces[i]['box']        
        else:
            x1, y1, width, height = 1, 1, 10, 10
            no_face = True
        
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = image_rgb[y1:y2, x1:x2]                        
        face = cv2.resize(face, (160, 160))
        face = asarray(face)
        
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        
        face = expand_dims(face, axis=0)
        signature = face_encoder.predict(face)
        
        min_dist=100
        identity=' '
        for key, value in database.items() :
            dist = np.linalg.norm(value-signature)
            if dist < min_dist:
                min_dist = dist
                identity = key
        print('minimum_dist: ', min_dist)
        if no_face: 
            identity = 'unknown'
            no_face = False
        #clean up identity
        identity = identity.replace('_', '')
        identity = re.sub(r'[0-9]', '', identity)
        identity = identity.split('.')[0]
        cv2.putText(image,identity, (x1-25,y1-100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(image,(x1,y1),(x2,y2), (0,255,0), 2)
        
    cv2.imshow('res',image)
    
    if cv2.waitKey(2) & 0XFF == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()