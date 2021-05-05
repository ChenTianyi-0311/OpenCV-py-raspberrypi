import cv2
import numpy as np


src = cv2.VideoCapture(0)
while(src.isOpened()):
    ret, frame = src.read()
    #hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    low_hsv=np.array([0,150,50])
    high_hsv=np.array([1,255,255])
    low_bgr = np.array([0,0,110])
    high_bgr = np.array([40,40,255])
    #low_hsv2=np.array([177,150,0])
    #high_hsv2=np.array([200,255,255])
    mask=cv2.inRange(frame,lowerb=low_bgr,upperb=high_bgr)
    #mask2=cv2.inRange(hsv,lowerb=low_hsv2,upperb=high_hsv2)
    #red=cv2.bitwise_or(mask,mask2)

    #对图像进行一些形态学操作把一些白色的小点去掉,把轮廓边缘变得更加清晰
    #高斯模糊
    blur=cv2.GaussianBlur(mask,(5,5),0)
    #二值处理
    thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #闭运算
    ker=np.ones((5,5),np.uint8)
    close=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,ker)
    #找到图像的轮廓
    contours,hierarchy=cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for i in contours:
        x,y,w,h=cv2.boundingRect(i)
        if  (240<y)&(y+h+w<400) :
            if w*h<500:
                pass
            else:
                img=frame[y:y+h,x:x+w]
            
                img=cv2.resize(img,(500,460))
                cv2.rectangle(frame,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
                print(w*h)
                if(w*h < 1000):
                    cv2.putText(frame, "Red light", (x,y+w+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                if(w*h >= 1000):
                    cv2.putText(frame, "Traffic sign", (x,y+w+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.line(frame, (0, 240), (640, 240), (0, 255, 0), 1, 4)
    cv2.line(frame, (0, 400), (640, 400), (0, 255, 0), 1, 4)        
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
src.release()

cv2.destroyAllWindows()
