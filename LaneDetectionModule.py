import cv2
import numpy as np
import utlis

curveList = []
avgVal = 10

def getLaneCurve(img,display):
    imgCopy = img.copy()
    imgResult = img.copy()
    # step 1
    imgThres = utlis.thresholding(img)

    # step 2
    hT, wT, c = img.shape
    points = utlis.valTrackbars()
    imgWarp = utlis.warpImg(imgThres, points, wT,hT)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)

    # step 3
    midPoint, imgHist = utlis.getHistogram(imgWarp,minPer=0.5, display=True,region=4)
    curveAveragePoint, imgHist = utlis.getHistogram(imgWarp,minPer=0.9, display=True,region=1)
    curveRaw = curveAveragePoint-midPoint

    # step 4
    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)

    curve = int(sum(curveList)/len(curveList))

    # step 5
    if display != 0:
        imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT,inv = True)
        # cv2.imshow("inv", imgInvWarp)
        imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT//3,0:wT] = 0,0,0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
        midY = 200
        cv2.putText(imgResult,str(curve),(wT//2-200,50),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,0,255),2)
        cv2.line(imgResult,(wT//2,midY),(wT//2+(curve*3),midY),(255,0,255),5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 0, 255), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                    (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        # cv2.putText(imgResult, 'FPS '+str(int(fps)), (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3)
    if display == 2:
        imgStacked = utlis.stackImages(0.8,([img,imgWarpPoints,imgWarp],
                                            [imgHist,imgLaneColor,imgResult]))
        cv2.imshow('ImageStack',imgStacked)
    elif display == 1:
        cv2.imshow('Resutlt',imgResult)
    
    # normalization
    if curve>-5 and curve<=5:
        curve = 0
    elif curve>-10 and curve<=-5:
        curve = -1
    elif curve<-10:
        curve = -2
    elif curve>5 and curve<=10:
        curve = 1
    elif curve>10:
        curve = 2
    return curve

if __name__ == '__main__':
    cap = cv2.VideoCapture('test_video_1.mp4')
    # cap = cv2.VideoCapture(0)
    intialTrackBarVals = [130,130,30,240]
    utlis.initializeTrackbars(intialTrackBarVals)
    frameCounter = 0
    
    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, img = cap.read()
        img = cv2.resize(img,(480,240))
        curve = getLaneCurve(img, display=2)
        print(curve)
        # cv2.imshow('video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break