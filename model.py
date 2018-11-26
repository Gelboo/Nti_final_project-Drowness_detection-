#
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from PIL import Image
#_______________******_________________*****___________________*****

'''
    this was for 2-D output layer
'''
# theLabels = np.array([[]],dtype='float')
# theImages = []
# # initalize with empty
# lines=[]
# num_points = 1
# with open('HelenDataSet/__Annotation/1new.txt','r') as file:
#     for line in file:
#         line = line.replace('\n','')
#         lines.append(line)
# # print(lines)
# # get the image title and remove the title from the lines
# img_title = lines.pop(0)
# label = np.array([[]],dtype='float')
# for item in lines:
#     values = item.split(',')
#     label = np.append(label,np.array(values,dtype='float'))
# label = label.reshape(num_points,2)
# theLabels = label
# theImages.append(img_title)
# # print(label)
# print(theLabels.shape)
#
# img = cv2.imread('HelenDataSet/{}.jpg'.format(img_title))
# new_img_width = 200
# new_img_height = 200
# #_______________*****________________*****___________________****
# # start from the second Image
# num_images = 73
# for i in range(2,num_images):
#     lines = []
#     file_path = "HelenDataSet/__Annotation/"+str(i)+"new.txt"
#     with open(file_path,'r') as file:
#         for line in file:
#             line = line.replace('\n','')
#             lines.append(line)
#     # print(lines)
#     # get the image title and remove the title from the lines
#     img_title = lines.pop(0)
#     label = np.array([[]],dtype='float')
#     for item in lines:
#         values = item.split(',')
#         label = np.append(label,np.array(values,dtype='float'))
#     label = label.reshape(num_points,2)
#     # print(theLabels.shape)
#     # print(label.shape)
#     theLabels = np.vstack((theLabels,label))
#     theImages.append(img_title)
#     # print(label)
#
#     img = cv2.imread('HelenDataSet/{}.jpg'.format(img_title))
#     new_img_width = 200
#     new_img_height = 200
#
#     # img = cv2.resize(img,(new_img_width,new_img_height))
#     # update label point
# theLabels = theLabels.reshape((num_images-1,num_points,2))
# print(theLabels.shape)
# # for lb in theLabels:
# #     print("start *** -___---")
# #     print(lb)
# #     print("end *** -+____--")
# # print(theLabels)
# # print(theImages)


'''
 make the label be flatten with shape (388,)
'''
theLabels = np.array([],dtype='float')
theImages = []
# initalize with empty
lines=[]
num_points = 194
with open('HelenDataSet/__Annotation/1new.txt','r') as file:
    for line in file:
        line = line.replace('\n','')
        lines.append(line)
# print(lines)
# get the image title and remove the title from the lines
img_title = lines.pop(0)
label = np.array([],dtype='float')
for item in lines:
    values = item.split(',')
    label = np.append(label,np.array(values,dtype='float'))
theLabels = label
theImages.append(img_title)
# print(label)
print(theLabels.shape)
# print(theLabels)

img = cv2.imread('HelenDataSet/{}.jpg'.format(img_title))
new_img_width = 200
new_img_height = 200
#_______________*****________________*****___________________****
# start from the second Image
num_images = 2330
for i in range(2,num_images):
    lines = []
    file_path = "HelenDataSet/__Annotation/"+str(i)+"new.txt"
    with open(file_path,'r') as file:
        for line in file:
            line = line.replace('\n','')
            lines.append(line)
    # print(lines)
    # get the image title and remove the title from the lines
    img_title = lines.pop(0)
    label = []
    for item in lines:
        values = item.split(',')
        label = np.append(label,np.array(values,dtype='float'))
    theLabels = np.append(theLabels,label)
    theImages.append(img_title)
    # print(label)
    # print(theLabels.shape)
    # print(theLabels)

    img = cv2.imread('HelenDataSet/{}.jpg'.format(img_title))
    new_img_width = 200
    new_img_height = 200

    # img = cv2.resize(img,(new_img_width,new_img_height))
    # update label point
theLabels = theLabels.reshape((num_images-1,num_points*2))
print("the Labels shape")
print(theLabels.shape)
# for lb in theLabels:
#     print("start *** -___---")
#     print(lb)
#     print("end *** -+____--")
# print(theLabels)
# print(theImages)


def prepare_X_Y():
    images = np.array([np.array(cv2.resize(cv2.imread('HelenDataSet/'+img+'.jpg'),(200,200))) for img in theImages])
    # print("imageeee")
    # print(images)
    x_y = np.array([[img,lbl] for img,lbl in zip(images,theLabels)])
    # print(x_y.shape)
    # print x_y[:,1]
    return x_y
x_y = prepare_X_Y()
# print(x_y_train.shape)
split = int(num_images*0.1)
x_y_test = x_y[:split]
x_y_train = x_y[split:]
print(x_y_train.shape)
print(x_y_test.shape)
for i in range(10):
    plt.subplot(2,5,i+1,xticks=[],yticks=[])
    plt.imshow(x_y_train[i,0][:,:,[2,1,0]],interpolation='nearest',aspect='auto')
plt.show()

def divide_img_lbl(data):
    """ split data into image and label"""
    x = []
    y = []
    for [item,lbl] in data:
        x.append(item)
        y.append(lbl) # change it to [lbl] if error in the model
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train,y_train = divide_img_lbl(x_y_train)

print(x_train.shape)
print(y_train.shape)

x_test,y_test = divide_img_lbl(x_y_test)

print(x_test.shape)
print(y_test.shape)
#

# rescale [0,255]  --> [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# # print x_train[0]
# bins = np.arange(0,36)
# lbl = fruitTypes
# import seaborn as sns
# sns.set()
# plt.hist(y_train,bins,ec='black')
# plt.xlabel('Labels')
# plt.ylabel('Frequency')
# plt.xticks(bins,lbl)
# plt.show()
import keras
#
# # one Hot_Encoding
# # print y_train
# num_classes = len(fruitTypes)
# # print len(y_train)
# # print y_train.max(),y_train.min()
# # print y_test.max(),y_test.min()
# # print num_classes
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
#
# break trainset into trainset and validationset
#take first 80% as train and 20% as validation
uptill = int(len(x_train)*0.8)
(x_train,x_valid) = x_train[:uptill],x_train[uptill:]
(y_train,y_valid) = y_train[:uptill],y_train[uptill:]
#
print("x_train_shape")
print(x_train.shape)
print(x_valid.shape)
print("y_train_shape")
print(y_train.shape)
print(y_valid.shape)
#
#
# # create the model
#
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
model2.add(Conv2D(filters = 16,kernel_size=2,padding='same',activation='relu',input_shape=(200,200,3)))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=128,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=256,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Conv2D(filters=512,kernel_size=2,padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=2))

model2.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(1000,activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(500,activation='relu'))
model2.add(Dropout(0.2))

model2.add(Dense(388,activation='relu'))
model2.summary()

#compile the model
model2.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='pointPredict.weights.best.hdf5', verbose=1,
                               save_best_only=True)
hist = model2.fit_generator(datagen.flow(x_train,y_train,batch_size=32),steps_per_epoch=x_train.shape[0]//32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)

# load weight with best validation score
model2.load_weights('pointPredict.weights.best.hdf5')

score = model2.evaluate(x_test,y_test,verbose=0)
print('test accuracy',score[1])
#
# Labels = fruitTypes
# print(Labels)
# y_hat = (model2.predict(x_test))
# for i in range(6):
#     plt.subplot(2,3,i+1,xticks=[],yticks=[])
#     plt.imshow(np.squeeze(x_test[i][:,:,[2,1,0]]))
#     pred_idx = np.argmax(y_hat[i])
#     true_idx = np.argmax(y_test[i])
#     plt.title("{} ({})".format(Labels[pred_idx],Labels[true_idx] ),color=("green" if pred_idx == true_idx else "red"))
# # plt.tight_layout()
# plt.show()
# '''
#
#
# #    Second Try...
#
#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# import glob
# import os
# def prepare_X_Y(path):
#     ApprovedList = glob.glob(path+'/Approved/*')
#     Approves = np.array([np.array(cv2.resize(cv2.imread(approve),(100,100))) for approve in ApprovedList])
#     print("number of the approved Images: ",Approves.shape[0])
#
#     x_y = np.array([[img,1] for img in Approves])
#     plt.suptitle('Approved Orange',fontsize=16,color='blue')
#     for i in range(10):
#         plt.subplot(5,2,i+1,xticks=[],yticks=[])
#         plt.imshow(Approves[i][:,:,[2,1,0]],interpolation='nearest' , aspect='auto')
#     plt.show()
#
#
#     defectedList = glob.glob(path+'/defected/*')
#     defects = np.array([np.array(cv2.resize(cv2.imread(defect),(100,100))) for defect in defectedList])
#     print("number of the defected Images: ",defects.shape[0])
#
#     x_y = np.vstack((x_y,np.array([[img,0] for img in defects])))
#     print("shape of dataSet: ",x_y.shape)
#     plt.suptitle('Defected Orange',fontsize=16,color='blue')
#     for i in range(10):
#         plt.subplot(5,2,i+1,xticks=[],yticks=[])
#         plt.imshow(defects[i][:,:,[2,1,0]],interpolation='nearest',aspect='auto')
#     plt.show()
#
#     np.random.shuffle(x_y)
#     plt.suptitle('Randomize Approved & Defected', fontsize=16,color='blue')
#     for i in range(10):
#         plt.subplot(5,2,i+1,xticks=[],yticks=[])
#         plt.imshow(x_y[i][0][:,:,[2,1,0]],interpolation='nearest',aspect='auto')
#     plt.show()
#
#     return x_y
#
# print("first Training .....")
# x_y_train = prepare_X_Y('Pictures_detection')
# print("size Of training: ",x_y_train.shape[0])
#
# print("second Testing ..... ")
# x_y_test = prepare_X_Y('Validation_detect')
# print('size of testing: ',x_y_test.shape[0])
#
# def divide_img_lbl(data):
#     #split data into image and label
#     x = []
#     y = []
#     for [item,lbl] in data:
#         x.append(item)
#         y.append(lbl)
#     x = np.array(x)
#     y = np.array(y)
#     return x,y
#
# x_train,y_train = divide_img_lbl(x_y_train)
# x_test,y_test = divide_img_lbl(x_y_test)
# print("train Input shape: ",x_train.shape)
# print("test Input shape: ",x_test.shape)
#
# x_train = x_train.astype('float32')/255
# x_test = x_test.astype('float32')/255
#
# uptill = int(len(x_train)*0.8)
# print(uptill)
# (x_train,x_valid) = x_train[:uptill],x_train[uptill:]
# (y_train,y_valid) = y_train[:uptill],y_train[uptill:]
# print("size of training Set: ",y_train.shape[0])
# print("size of validation set: ",y_valid.shape[0])
#
# import keras
# from keras.models import Sequential
# from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
# from keras.preprocessing.image import ImageDataGenerator
#
# # create and configure augmented image generator
# datagen = ImageDataGenerator(
#             width_shift_range=0.1,
#             height_shift_range=0.1,
#             horizontal_flip=True)
#
# model = Sequential()
# model.add(Conv2D(filters=16,kernel_size=2,padding='same',activation='relu',input_shape=(100,100,3)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=12,kernel_size=2,padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(500,activation='relu'))
# model.add(Dropout(0.4))
#
# model.add(Dense(1,activation='sigmoid'))
# model.summary()
#
# # compile the model
# model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])
#
# from keras.callbacks import ModelCheckpoint
# checkpointer = ModelCheckpoint(filepath='modelDetectOrange2.weights.best.hdf5',verbose=1,save_best_only=True)
#
# hist = model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),steps_per_epoch=x_train.shape[0]//32,epochs=100,validation_data=(x_valid,y_valid),callbacks=[checkpointer],verbose=2,shuffle=True)
#
# # load weight with best validation score
# model.load_weights('modelDetectOrange2.weights.best.hdf5')
#
# score = model.evaluate(x_test,y_test,verbose=0)
# print('test accuracy',score[1])
#
# Labels = ['Defected','Approved']
# y_hat = model.predict(x_test)
# y_hat = np.round(y_hat).astype('int').flatten()
# for i in range(10):
#     plt.subplot(5,2,i+1,xticks=[],yticks=[])
#     plt.imshow(x_test[i][:,:,[2,1,0]],interpolation='nearest',aspect='auto')
#     plt.title("{} ({})".format(Labels[y_hat[i]],Labels[y_test[i]] ),color=("green" if y_hat[i] == y_test[i] else "red"))
# # plt.tight_layout()
# plt.show()
# '''
