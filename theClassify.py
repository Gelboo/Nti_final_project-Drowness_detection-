import cv2

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

model2 = Sequential()
model2.add(Conv2D(filters = 16,kernel_size=2,padding='same',activation='relu',input_shape=(100,100,3)))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))


model2.add(Dropout(0.2))
model2.add(Flatten())

model2.add(Dense(500,activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(500,activation='relu'))
model2.add(Dropout(0.2))

model2.add(Dense(2,activation='sigmoid'))
# model2.summary()

Labels_left = ['openLeftEyes','closedLeftEyes']
Labels_right = ['openRightEyes','closedRightEyes']
def get_status(img,pos):
    if pos == 'left_eye':
        model2.load_weights('EyeStateLeft.weights.best.hdf5')
        Labels = Labels_left
    elif pos == 'right_eye':
        model2.load_weights('EyeStateRight.weights.best.hdf5')
        Labels = Labels_right
    img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(100,100))
    img = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
    y_hat = (model2.predict(img))
    id = np.argmax(y_hat)
    return(Labels[id])

if __name__ == '__main__':
    Labels_left = ['closedLeftEyes', 'openLeftEyes']
    Labels_right = ['closedRightEyes', 'openRightEyes']
    # img = cv2.imread('Alicia_Witt_0001_L.jpg')
    # img = cv2.imread('Ally_Sheedy_0001_L.jpg')
    # img =cv2.imread('Ali_Hammoud_0001_L.jpg')
    img = cv2.imread('Abdel_Madi_Shabneh_0001_L.jpg')
    # img = cv2.imread('closed_eye_0061.jpg_face_1_L.jpg')
    # img = cv2.imread('closed_eye_0038.jpg_face_1_L.jpg')
    #
    #
    # img = cv2.imread('Alberto_Gonzales_0001_L.jpg')
    #
    print(get_status(img,'left_eye'))
    img = cv2.resize(img,(400,200))
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
