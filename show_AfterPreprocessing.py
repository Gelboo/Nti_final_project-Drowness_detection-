import cv2
import numpy as np

import pandas as pd


for i in range(1,2331):
    lines = []
    file_path = "HelenDataSet/__Annotation/"+str(i)+"new.txt"
    with open(file_path,'r') as file:
        for line in file:
            line = line.replace('\n','')
            lines.append(line)
    # print(lines)
    # get the image title and remove the title from the lines
    img_title = lines.pop(0)
    label = np.array([[]],dtype='float')
    for item in lines:
        values = item.split(',')
        label = np.append(label,np.array(values,dtype='float'))
    label = label.reshape(194,2)
    # print(label)

    img = cv2.imread('HelenDataSet/{}.jpg'.format(img_title))
    new_img_width = 200
    new_img_height = 200

    img = cv2.resize(img,(new_img_width,new_img_height))
    # update label point

    for p in label:
        cv2.circle(img,(int(p[0]),int(p[1])),3,(255,0,0),-1)

    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
