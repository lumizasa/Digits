from imutils.perspective import four_point_transform
from imutils import contours
import imutils
from time import sleep
from datetime import date
import numpy as np
import cv2
import tensorflow as tf
import serial
nrec= tf.keras.models.load_model('ReconocerDigitos300')
cam=cv2.VideoCapture(1)

kernel = np.ones((5,5),np.uint8)
lowhsv=np.array([0,0,86])
highhsv=np.array([255,104,255])

today = date.today()
d1 = today.strftime("%d%m%Y")

#print(d1)

ard = serial.Serial('com7', 9600)
sleep(0.1)

while True:
    hMin = 5 #cv2.getTrackbarPos('H Min','image_5')
    hMax = 40 #cv2.getTrackbarPos('H Max','image_5')
    wMin = 5 #cv2.getTrackbarPos('W Min','image_5')
    wMax = 40 #cv2.getTrackbarPos('W Max','image_5')

    _,image = cam.read()
    
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    thsv=cv2.inRange(hsv,lowhsv,highhsv)## valores hsv

    cnts = cv2.findContours(thsv.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    hoja = None
    
    # loop over the contours
    for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                    hoja = approx
                    break
    cv2.drawContours(image, hoja, -1, (0,250,0), 5)
    cv2.imshow('video',image)
    if type(hoja) == None:
        print('sin hojas')
        ard.write(b'3')
        sleep(0.5)
    else:
        try:
            #warped = four_point_transform(thsv, hoja.reshape(4, 2)) 
            output = four_point_transform(image, hoja.reshape(4, 2))
            gray=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
            #cv2.imshow('Hoja en gris',gray)
            warped=cv2.inRange(gray,0,158)                                                          ### Valores gris
            cv2.imshow('Mascara en la hoja',warped)
        
            cnts = cv2.findContours(warped.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            digitCnts = []
            # loop over the digit area candidates
            for c in cnts:
                # compute the bounding box of the contour
                (x, y, w, h) = cv2.boundingRect(c)
                # if the contour is sufficiently large, it must be a digit
                if (w > wMin and w < wMax) and (h > hMin and h < hMax):
                    digitCnts.append(c)
            digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]
            cv2.drawContours(output,digitCnts,-1,(255,0,0),3)
            print('Numeros reconocidos: ',len(digitCnts))
            cv2.imshow('Salida',output)
            if len(digitCnts)==8:
                digits = []
                #Loop over each contours looking for digits
                #a=0
                for c in digitCnts:
                    #a=a+1
                    (x,y,w,h)=cv2.boundingRect(c)
                    roi=warped[y-3:y+h+3 , x-3:x+w+3]
                    roi=cv2.resize(roi,(22,45))
                    #cv2.imshow('roi'+str(a),roi)
                    digits.append(roi)

                numero=""
                npdigits=np.array(digits)
                prd=nrec.predict(npdigits)
    ##            for x in range(len(prd)):
    ##                print('creo que es: ',np.argmax(prd[x]))
    ##                cv2.imshow('roi',digits[x])
    ##                cv2.waitKey(0)
                for x in range(len(prd)):
                    #numero.append(np.argmax(prd[x]))
                    numero=numero+str(np.argmax(prd[x]))
                    
                #print('El numero es: ',numero) 

                if numero == d1:
                    print('Fecha Correcta')
                    ard.write(b'0')
                    sleep(1)
                else:
                    print('Fecha Incorrecta')
                    ard.write(b'1')


            
        except: 
            pass
    if cv2.waitKey(5)==ord('q'):
        break
    sleep(0.2)


cv2.destroyAllWindows()
cam.release()

