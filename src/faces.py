# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pickle
from keras.preprocessing.image import img_to_array
from keras.models import load_model

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

detection_model_path = 'cascades/data/haarcascade_frontalface_alt2.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}


cap = cv2.VideoCapture(0)
#cap.set(3, 600)
#cap.set(4, 800)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
            '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe(
    'deploy_gender.prototxt', 'gender_net.caffemodel')


while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for(x, y, w, h) in faces:
      
        roi_gray = gray[y:y + h, x:x + w]
      

        id_, conf = recognizer.predict(roi_gray)
        #print(conf, id_)
        if conf >= 65: 
           font = cv2.FONT_HERSHEY_SIMPLEX
           name = labels[id_]
           color = (255, 255, 255)
           stroke = 1
           cv2.putText(frame, name, (x, y), font, 1,
                       color, stroke, cv2.LINE_AA)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Get Face
        face_img = frame[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(
            face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
      

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
     
        font = cv2.FONT_HERSHEY_SIMPLEX
        overlay_text = "%s %s" % (gender, age)
        cv2.putText(frame, overlay_text, (end_cord_x, end_cord_y),
                    font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        

        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[:y + h, x:x + w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        
        cv2.putText(frame, label, (end_cord_x, end_cord_y - 40 ),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
