import cv2
from cascade import get_parts

v = cv2.VideoCapture(0)
while True:
    _,frame = v.read()
    img,_,_ = get_parts(frame)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
