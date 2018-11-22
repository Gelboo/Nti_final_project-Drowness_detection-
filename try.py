import numpy as np
import cv2
from numpy import newaxis

# a = np.array([[1,2,4],[5,6,7]])
# print(a.shape)
#
# b = a[newaxis,:,:]



# v = np.array(
# [
#     [
#         [1,2,3],
#         [4,5,6]
#     ],
#     [
#         [5,6,7],
#         [8,9,10]
#     ]
# ])
# print(v.shape)
# print(b.shape)


# a = np.array([[]],dtype='float')
# img1 = np.array([[40,30],[50,80],[12,10],[50,80],[50,80]])
# img2 = np.array([[96,130],[50,80],[93,10],[90,80],[50,80]])
# img3 = np.array([[30,60],[50,90],[12,14],[50,80],[50,80]])
# print(img1.shape)
# a = np.vstack((img1,img2,img3))
# a = a.reshape((3,5,2))
# print(a)

img = cv2.imread('HelenDataSet/104074861_1.jpg')
print(img.shape)
