import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haaracascade_lefteye_2splits.xml')


def get_parts(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    roi_faces = []
    roi_eyes = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w].copy()
        roi_face_color = img[y:y+h, x:x+w]
        roi_faces.append(face_color)
    if roi_faces != []:
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            roi = roi_face_color[ey:ey+eh, ex:ex+ew]
            cv2.rectangle(roi_face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            roi_eyes.append(roi)
    return img,roi_faces,roi_eyes
