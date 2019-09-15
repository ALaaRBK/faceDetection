import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def smileDetection(grey_img,frame):
    path = os.path.dirname(os.path.abspath(__file__)) 
    smile_cascade = cv.CascadeClassifier(path + '/data/haarcascade_smile.xml')
    smile_detect = smile_cascade.detectMultiScale(grey_img,1.8,20)
    for (x,y,w,h) in smile_detect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv.putText(frame,"smile",(x,y),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,0))

    return frame



if __name__ == "__main__":
    from detections.utils import detectionAll
    detectionAll("smile")