import cv2
import numpy as np

import pandas as pd


for i in range(1,2331):
    lines = []
    file_path = "HelenDataSet/annotation/"+str(i)+".txt"
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
    # If you required to resize the image I'll need to update the label for the new size
    # get the original image size
    # print(img.shape)
    img_old_height = img.shape[0]
    img_old_width = img.shape[1]
    #
    new_img_width = 200
    new_img_height = 200

    img = cv2.resize(img,(new_img_width,new_img_height))
    # update label point

    with open('HelenDataSet/__Annotation/'+str(i)+'new.txt','w') as file:
        file.writelines(img_title+'\n')
        for p in label:
            p[0] = (new_img_width/img_old_width)*p[0]
            p[1] = (new_img_height/img_old_height)*p[1]
            file.writelines(str(p[0])+','+str(p[1])+'\n')
            # print(p)
            cv2.circle(img,(int(p[0]),int(p[1])),3,(255,0,0),-1)
    # cv2.imshow("image",img)
    # cv2.waitKey(10)
    # cv2.destroyAllWindows()
