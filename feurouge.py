import cv2 as cv
import numpy as np

a=0
src = cv.VideoCapture(0)
while(src.isOpened()):
    ret, frame = src.read()
    #height = np.size(frame,0)
    #width = np.size(frame,1)
    #print(height,width)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    low_hsv=np.array([0,80,50])
    high_hsv=np.array([3,255,220])
    low_hsv2=np.array([170,80,50])
    high_hsv2=np.array([180,255,220])
    mask=cv.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    mask2=cv.inRange(hsv,lowerb=low_hsv2,upperb=high_hsv2)
    #print(type(mask))
    red=cv.bitwise_or(mask,mask2)

    blur=cv.GaussianBlur(red,(5,5),0)

    thresh=cv.threshold(blur,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    ker=np.ones((5,5),np.uint8)
    close=cv.morphologyEx(thresh,cv.MORPH_CLOSE,ker)

    contours,hierarchy=cv.findContours(close,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

    for i in contours:
        x,y,w,h=cv.boundingRect(i)
        if  (240<y)&(y+h+w<400) :
            if w*h<200:
                pass
            else:
                a+=1
                img=frame[y:y+h,x:x+w]
            
                img=cv.resize(img,(500,460))
                cv.rectangle(frame,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
                print(w*h)
                if(w*h < 400):
                    cv.putText(frame, "Red light", (x,y+w+10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                if(w*h >= 400):
                    cv.putText(frame, "Traffic sign", (x,y+w+10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, thresh = cv.threshold(gray, 70, 255, cv.THRESH_BINARY_INV)

                ker = np.ones((6, 6), np.uint8)
                close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, ker)

                #h, w = gray.shape[0], gray.shape[1]
                #point1 = [0.15 * w, h / 4]
                #point2 = [0.15 * w, 4 * h / 5]
                #point3 = [0.83 * w, 4 * h / 5]
                #point4 = [0.83 * w, h / 4]
                #list1 = np.array([[point1, point2, point3, point4]], dtype=np.int32)
                #mask = np.zeros_like(gray)
               # mask = cv.fillConvexPoly(mask, list1, 255)
    cv.line(frame, (0, 240), (640, 240), (0, 255, 0), 1, 4)
    cv.line(frame, (0, 400), (640, 400), (0, 255, 0), 1, 4)        
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
            break
src.release()

cv.destroyAllWindows()