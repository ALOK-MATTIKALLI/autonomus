import cv2
from motorModule import Motor
from LaneDetectionModule import getLaneCurve
import WebcamModule
import time
 
##################################################
motor = Motor(10,9,17,27)
##################################################
 
def main():
 
    img = WebcamModule.getImg(display=True)
    # cap = cv2.VideoCapture(0)
    # while True:
    #     success, img = cap.read()
    #     img = cv2.resize(img,(480,240))
    #     curve = getLaneCurve(img, display=2)
    #     print(curve)
    #     # cv2.imshow('video', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    curveVal= getLaneCurve(img,display=4)
    time.sleep(2.0)
    # ..........................
    # sen = 1.3  # SENSITIVITY
    # maxVAl= 0.3 # MAX SPEED
    # if curveVal>maxVAl:curveVal = maxVAl
    # if curveVal<-maxVAl: curveVal =-maxVAl
    # #print(curveVal)
    # if curveVal>0:
    #     sen =0.7
    #     if curveVal<0.05: curveVal=0
    # else:
    #     if curveVal>-0.08: curveVal=0
    
    # motor.move(0.20,-curveVal*sen,0.0001)
    # ..............................
    motor.move(0.02,-curveVal,0.0001)
    # print(curveVal)
    cv2.waitKey(1)
    
     
 
if __name__ == '__main__':
    while True:
        main()
motor.stop(0.1)