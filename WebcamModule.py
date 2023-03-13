import cv2
 
cap = cv2.VideoCapture(0)
 
def getImg(display=False,size=[320,240]):
    _, img = cap.read()
    img = cv2.resize(img,(size[0],size[1]))
    if display == True:
        cv2.imshow('IMG',img)
        
    return img
 
if __name__ == '__main__':
    while True:
        img = getImg(display=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break