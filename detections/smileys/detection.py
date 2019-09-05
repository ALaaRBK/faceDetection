import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


def smileDetection(grey_img,frame):
    path = os.path.dirname(os.path.abspath(__file__)) 
    smile_cascade = cv.CascadeClassifier(path + '/data/haarcascade_smile.xml')
    smile_detect = smile_cascade.detectMultiScale(grey_img,scaleFactor=1.1,minNeighbors=12)
    text = 'number Of eyes =' + str(len(smile_detect)) 
    cv.putText(frame,text,(10,100),cv.FONT_HERSHEY_COMPLEX,1,(255,255,0))
    for (x,y,w,h) in smile_detect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv.putText(frame,"eye",(x,y),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,0))

    return frame