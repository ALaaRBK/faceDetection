import cv2 as cv
import os
import sys
from detections.faces.detection import faceDetection
from detections.eyes.detection import eyesDetection
from detections.smileys.detection import smileDetection

def detectionAll(*argv):
    print(argv[0])
    cap = cv.VideoCapture(0)

    while True :
        ret,frame = cap.read()
        if ret == True:
            grey_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            path = os.path.dirname(os.path.abspath(__file__)) 
            cv.putText(frame,"PRESS Q TO EXIT",(10,150),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,0))
            arguments = len(argv[0]) - 1
            position = 1
            while (arguments >= position):
                frame = globals()[argv[0][position] + "Detection"](grey_img,frame)
                position = position + 1
            
            cv.imshow('frame',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break


    cap.release()
    cv.destroyAllWindows()

