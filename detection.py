import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
# C:/Users/12029/venv/Scripts/python c:/Users/12029/Documents/WhiteHatProjects/DuringClassProjectd/c123/digitRecogniton.py
from sklearn.model_selection import train_test_split as tts 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as ass
from PIL import Image
import PIL.ImageOps
import os,ssl,time 
import cv2  

X,y=fetch_openml("mnist_784",version=1,return_X_y=True)

classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nClasses=len(classes)

xTrain,xTest,yTrain,yTest=tts(X,y,train_size=10000,test_size=3000,random_state=9)

xTrainScaled=xTrain/255
xTestScale=xTest/255

model=LogisticRegression(solver="saga",multi_class="multinomial").fit(xTrainScaled,yTrain)

yPredict=model.predict(xTest)
accuracy=ass(yTest,yPredict)

print(accuracy)

cam=cv2.VideoCapture(0)

while True:
    #capture frame by frame 
    try:
        ret,frame=cam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))

        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        iampill=Image.fromarray(roi)
        imageBW=iampill.convert("L")
        #this image will be represented by a single from 0 to 255
        imageBWResized=imageBW.resize((28,28),Image.ANTIALIAS)
        imageInverted=PIL.ImageOps.invert(imageBWResized)
        pixelFilter=20
        minPixel=np.percentile(imageInverted,pixelFilter)

        finalImage=np.clip(imageInverted-minPixel,0,255)
        maxPixel=np.max(imageInverted)

        finalImage=np.asarray(finalImage)/maxPixel

        testSample=np.array(finalImage).reshape(1,784)
        testPredict=model.predict(testSample)

        print(testSample)
        cv2.imshow("frame",gray)

    except Exception as e:
        pass


cam.release()
cv2.destroyAllWindows()