import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog

theta = np.loadtxt('theta.txt')

img=cv2.imread('D:\Desktop\OpenCv lib\so1.jpg')
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray=cv2.GaussianBlur(img_gray, (5,5), 0)
_,im_th=cv2.threshold(img_gray, 155, 255, cv2.THRESH_BINARY_INV)
ctrs,_=cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects=[cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:
    cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255),3)

    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
    roi = cv2.resize(roi, (28, 28), im_th, interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    number=np.array([roi]).reshape(1,28*28)
    # predict = model.predict(number)
    one = np.ones((number.shape[0], 1))
    number = np.concatenate((number, one), axis=1)
    predict = 1.0/ (1+np.exp(-np.dot(number, theta.T)))
    print('prediction', str(int(predict[0])))
    print(predict)
    cv2.putText(img,str(int(predict[0])),(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1.5,(165,42,42),2)

cv2.imshow('img', img)
cv2.imshow('img_gray', img_gray)
cv2.imshow('im_th', im_th)
cv2.waitKey(0)
cv2.destroyAllWindows()

