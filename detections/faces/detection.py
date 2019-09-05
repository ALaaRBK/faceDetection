import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from detections.eyes.detection import eyesDetection
from detections.smileys.detection import smileDetection
def FaceDetection():
    cap = cv.VideoCapture(0)

    while True :
        ret,frame = cap.read()
        if ret == True:
            grey_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            path = os.path.dirname(os.path.abspath(__file__)) 
            face_cascade = cv.CascadeClassifier(path+'/data/haarcascade_frontalface_alt.xml')

            face_detect = face_cascade.detectMultiScale(grey_img,1.1,minNeighbors=12)
            text  = 'number Of Faces =' + str(len(face_detect))
            cv.putText(frame,text,(10,50),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0))
            cv.putText(frame,"PRESS Q TO EXIT",(10,150),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,0))

            for (x,y,w,h) in face_detect:
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv.putText(frame,"face",(x,y),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,0))

            frame = eyesDetection(grey_img,frame)
            frame = smileDetection(grey_img,frame)
   
            cv.imshow('frame',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break


    cap.release()
    cv.destroyAllWindows()