import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


def eyesDetection(grey_img,frame):
    path = os.path.dirname(os.path.abspath(__file__)) 
    eyes_cascade = cv.CascadeClassifier(path + '/data/haarcascade_eye_tree_eyeglasses.xml')
    eyes_dedect = eyes_cascade.detectMultiScale(grey_img,scaleFactor=1.05,minSize=(20,20))
    text = 'number Of eyes =' + str(len(eyes_dedect)) 
    cv.putText(frame,text,(10,100),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0))
    for (x,y,w,h) in eyes_dedect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv.putText(frame,"eye",(x,y),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,0))

    return frame

if __name__ == "__main__":
    from detections.utils import detectionAll
    detectionAll("eyes")