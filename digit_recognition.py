import cv2
import numpy as np
import pandas  as pd
import seaborn as sns
import matplotlib as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time


if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl,'_create_unverified_context',None)):
 ssl._create_default_https_context=ssl._create_unverified_context

X,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print(pd.Series(y).value_counts())

classes=['0','1','2','3','4','5','6','7','8','9']
Nclases=len(classes)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0

clf=LogisticRegression(solver-'saga',multi_class='multinomial').fit(X_train_scaled,Y_train)

Y_predict=clf.predict(X_test_scaled)
accuracy=accuracy_score(Y_test,Y_predict)
print(accuracy)

#startingthe camera

cap=cv2.VideoCapture(0)

while(True):
    try:
        ret,frame=cap.read() 
        #converting the video into a grayscale video
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #drawing the box in teh  centre of the video
        height,width=gray.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        #consider the area inside the box for considering the digit
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]
        #comverting_the image
        im_PIL=Image.fromarray(roi)
        image_bw=im_PIL.convert('L')
        image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)
        #the image has to be inverted as it will apperar inverted after resizing
        image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
        pixel_filter=20
        min_pixel=np.percentile(image_bw_resized_inverted,pixel_filter)
        image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-min_pixel,0,255)
        max_pixel=np.max(image_bw_resized_inverted)
        #you will able to test the precdiction
        image_bw_resized_inverted_scaled=np.as_array(image_bw_resized_inverted_scaled)/max_pixel
        test_sample=np.array(image_bw_resized_inverted_scaled).reashape(1,784)
        test_predoct-clf.predoct(test_sample)
        print(test_predict)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()