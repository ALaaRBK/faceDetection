import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
def faceDetection(grey_img,frame):
    path = os.path.dirname(os.path.abspath(__file__)) 
    face_cascade = cv.CascadeClassifier(path+'/data/haarcascade_frontalface_alt.xml')
    face_detect = face_cascade.detectMultiScale(grey_img,1.1,minNeighbors=12)
    text  = 'number Of Faces =' + str(len(face_detect))
    cv.putText(frame,text,(10,50),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0))

    for (x,y,w,h) in face_detect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv.putText(frame,"face",(x,y),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,0))
    
    return frame



if __name__ == "__main__":
    from detections.utils import detectionAll
    detectionAll("face")