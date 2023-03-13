import cv2
import numpy as np


def canny_cvt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2.0, 2.0, 0)
    canny = cv2.Canny(blur, 5, 8)
    return canny

def n_canny_cvt(image):
    n_canny_img = cv2.Canny(image, 200, 255)
    return n_canny_img

def roi(image):
    height = image.shape[0]
    width = image.shape[1]
    poly = np.array([
        [(1, height-1), (width-1, height-1), 
         (int(width*0.90), int(height*0.2)), (int(width*0.10), int(height*0.2))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, poly, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

def thresh(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, t_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(gray, (7, 7), 2.0, 0, 2.0)
    ret, t_img = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return t_img

def prespective_trans(image,inv=False):
    height = image.shape[0]
    width = image.shape[1]
    poi = np.float32([[0, int(height*(0.5))],
                      [width-1, int(height*(0.5))],
                      [0, height-1],
                      [width-1, height-1],])
    proj = np.float32([[0, 0],
                       [width-1, 0],
                       [0, height-1],
                       [width-1, height-1]])
    m = cv2.getPerspectiveTransform(poi, proj)
    m_inv = cv2.getPerspectiveTransform(proj, poi)
    if (inv == False):
        w_img = cv2.warpPerspective(image, m, (width, height))
    else:
        w_img = cv2.warpPerspective(image, m_inv, (width, height))
    return w_img

def contrast(thresh_img,warp_img, image):
    contours, hierarchies = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # blank = np.zeros(thresh_img.shape[:2],dtype='uint8')
    blank = np.zeros_like(thresh_img)
    blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
    full = np.full(thresh_img.shape[:2],255, dtype='uint8')
    full = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(blank, contours, -1,(255, 0, 0), 1)
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(blank, [i], -1, (0, 255, 0), 2)
            cv2.circle(blank, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(blank, "center", (cx - 20, cy - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        print(f"x: {cx} y: {cy}")
    blank = prespective_trans(blank,inv=True)
    combine = cv2.addWeighted(image, 0.9, blank, 1, 0)
    return blank,combine, cx, cy

img = cv2.imread('/home/pi/autonomus_robo/123.png')
lane_img = np.copy(img)
warp = prespective_trans(lane_img)
th_img = thresh(warp)
cent, combo, x, y = contrast(th_img,warp, img)
# cv2.imshow('orig', img)
cv2.imshow('thresh', th_img)
# cv2.imshow('cent', cent)
cv2.imshow('combine', combo)
cv2.waitKey(0)

# ............................................................

# cap = cv2.VideoCapture('/home/alok/new_lane.py/123_t.mp4')
# while cap.isOpened:
#     _, frame = cap.read()
#     lane_img = np.copy(frame)
#     warp = prespective_trans(lane_img)
#     th_img = thresh(warp)
#     cent, combo = contrast(th_img,warp, frame)
#     # cv2.imshow('orig', img)
#     # cv2.imshow('thresh', th_img)
#     # cv2.imshow('cent', cent)
#     cv2.imshow('combine', combo)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break