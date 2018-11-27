
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import sys

os.chdir('dataset_B_Eye_Images/Left/')
EyeState = os.listdir(".")
print(EyeState)
os.chdir('../..')

def prepare_X_Y(path):
    lbl = 0
    # print(path+EyeState[0])
    EyeList = glob.glob(path+EyeState[0]+'/*')
    # print(EyeList)
    print('\n\n')
    Eyes = np.array([np.array(cv2.resize(cv2.imread(Eye),(100,100))) for Eye in EyeList])
    x_y = np.array([[img,lbl] for img in Eyes])
    print("x_y shape before append [first folder] =>",x_y.shape)
    lbl += 1
    rest = EyeState[1:]
    print(rest)
    for f in rest:
        # print(f)
        EyeList = glob.glob(path + f + '/*')
        # print(EyeList)
        cv2.destroyAllWindows()
        Eyes = np.array([np.array(cv2.resize(cv2.imread(Eye),(100,100))) for Eye in EyeList])
        x_y = np.vstack((x_y,np.array([[img,lbl] for img in Eyes])))
        print("x_y shape after append every folder => ",x_y.shape)
        lbl += 1
    print("",x_y.shape)
    # print x_y[:,1]
    np.random.shuffle(x_y)
    return x_y
x_y_data = prepare_X_Y('dataset_B_Eye_Images/Left/')
print(x_y_data.shape)


for i in range(36):
    plt.subplot(6,6,i+1,xticks=[],yticks=[])
    plt.imshow(x_y_data[i,0][:,:,[2,1,0]],interpolation='nearest',aspect='auto')
plt.show()

split = int(x_y_data.shape[0]*0.2)
print("split ",split)

x_y_test = x_y_data[:split]
x_y_train = x_y_data[split:]

print("train_size ",x_y_train.shape)
print("test_size ",x_y_test.shape)

def divide_img_lbl(data):
    """ split data into image and label"""
    x = []
    y = []
    for [item,lbl] in data:
        x.append(item)
        y.append([lbl])
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train,y_train = divide_img_lbl(x_y_train)

print("the input(train) size to the model")
print(x_train.shape)
print("the output(train) size to the model")
print(y_train.shape)

x_test,y_test = divide_img_lbl(x_y_test)
print("the input(test) size to the model")
print(x_test.shape)
print("the output(test) size to the model")
print(y_test.shape)
# rescale [0,255]  --> [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
y_train = y_train
y_test = y_test
# print x_train[0]
bins = np.arange(0,36)
lbl = EyeState
import seaborn as sns
sns.set()
plt.hist(y_train,bins,ec='black')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.xticks(bins,lbl)
plt.show()


import keras

# one Hot_Encoding
# print y_train
num_classes = len(EyeState)
# print len(y_train)
# print y_train.max(),y_train.min()
# print y_test.max(),y_test.min()
# print num_classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# break trainset into trainset and validationset
#take first 80% as train and 20% as validation
uptill = int(len(x_train)*0.8)
(x_train,x_valid) = x_train[:uptill],x_train[uptill:]
(y_train,y_valid) = y_train[:uptill],y_train[uptill:]

print("x train shape")
print(x_train.shape)
print(x_valid.shape)
print("y train shape")
print(y_train.shape)
print(y_valid.shape)


# create the model

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

from keras.preprocessing.image import ImageDataGenerator
# create and configure augmented image generator
datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

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
model2.summary()
#compile the model
model2.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='EyeStateLeft.weights.best.hdf5', verbose=1,
                               save_best_only=True)
hist = model2.fit_generator(datagen.flow(x_train,y_train,batch_size=32),steps_per_epoch=x_train.shape[0]//32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)

# load weight with best validation score
model2.load_weights('EyeStateLeft.weights.best.hdf5')

score = model2.evaluate(x_test,y_test,verbose=0)
print('test accuracy',score[1])

Labels = EyeState
print(Labels)
y_hat = (model2.predict(x_test))
for i in range(6):
    plt.subplot(2,3,i+1,xticks=[],yticks=[])
    plt.imshow(np.squeeze(x_test[i][:,:,[2,1,0]]))
    pred_idx = np.argmax(y_hat[i])
    true_idx = np.argmax(y_test[i])
    plt.title("{} ({})".format(Labels[pred_idx],Labels[true_idx] ),color=("green" if pred_idx == true_idx else "red"))
# plt.tight_layout()
plt.show()
