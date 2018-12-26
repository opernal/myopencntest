import cv2

cap=cv2.VideoCapture('http://192.168.0.52:8080/?action=stream')

while True:
        ret, frame = cap.read()
        cv2.imshow('video', frame)
        if cv2.waitKey(1)>0:
            exit(0)