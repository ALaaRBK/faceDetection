from detections.utils import detectionAll
from detections.faces.detection import faceDetection
from detections.eyes.detection import eyesDetection
from detections.smileys.detection import smileDetection
import sys

def main():
    arguments = len(sys.argv) - 1
    position = 1
    while (arguments >= position):
        globals()[sys.argv[arguments]]()
        position = position + 1
    # FaceDetection()

if __name__ == '__main__':
    detectionAll(sys.argv) 