import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time



face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haar cascade files\haarcascade_eye.xml')
mouth = cv2.CascadeClassifier('haar cascade files\haarcascade_smile.xml')
nose = cv2.CascadeClassifier('haar cascade files\haarcascade_mcs_nose.xml')

model = load_model('models/cnncat2.h5')
path = os.getcwd()
# cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
# ds_factor = 0.5
if mouth.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')
# To detect face and eyes in static images comment line-16, rewrite line-24 as "img = cv2.imread("imagepath")" 
while(True):
    # ret, img = cap.read()
    img = cv2.imread("image.jpg")
    # outpath = "image1.jpg"
    # cv2.imwrite(outpath,img,[int(cv2.IMWRITE_JPEG_QUALITY), 30])
    height,width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.3,5)
    count=count+1
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,'Face',(x,y-4),font,1,(0,0,255),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(roi_gray,1.3,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
            cv2.putText(roi_color,'Eyes',(ex,ey-4),font,0.75,(0,0,255),1)
        nose_rects = nose.detectMultiScale(roi_gray,1.3,5)
        for (ex,ey,ew,eh) in nose_rects:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 1)
            cv2.putText(roi_color,'Nose',(ex,ey-4),font,0.75,(0,0,255),1)
        mouth_rects = mouth.detectMultiScale(roi_gray,1.8,20)
        for (ex,ey,ew,eh) in mouth_rects:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 1)
            cv2.putText(roi_color,'Mouth',(ex,ey-4),font,0.75,(0,0,255),1)
    # cv2.putText(img,'Timer:'+str(count),(10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
    cv2.imshow('Face Detection',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
