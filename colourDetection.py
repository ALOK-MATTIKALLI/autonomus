import cv2
import numpy as np

def nothing():
    pass

cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars',600,250)
cv2.createTrackbar('Huemin', 'Trackbars', 0,180, nothing)
cv2.createTrackbar('Huemax', 'Trackbars', 180,180, nothing)
cv2.createTrackbar('Satmin', 'Trackbars', 0,255, nothing)
cv2.createTrackbar('Satmax', 'Trackbars', 255,255, nothing)
cv2.createTrackbar('Valmin', 'Trackbars', 0,255, nothing)
cv2.createTrackbar('Valmax', 'Trackbars', 255,255, nothing)

while True:
    img = cv2.imread('test_img.webp')
    hmin = cv2.getTrackbarPos('Huemin','Trackbars')
    hmax = cv2.getTrackbarPos('Huemax','Trackbars')
    smin = cv2.getTrackbarPos('Satmin','Trackbars')
    smax = cv2.getTrackbarPos('Satmax','Trackbars')
    vmin = cv2.getTrackbarPos('Valmin','Trackbars')
    vmax = cv2.getTrackbarPos('Valmax','Trackbars')
    print(hmin,hmax,smin,smax,vmin,vmax)
    lower = np.array([hmin,smin,vmin])
    upper = np.array([hmax,smax,vmax])
    mask = cv2.inRange(img, lower, upper)
    cv2.imshow('mask', mask)
    cv2.imshow('img', img)
    cv2.waitKey(1)


cv2.destroyAllWindow()