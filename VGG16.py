import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path = 'Path'
test_path = 'Path'

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(100,100),classes=['positive','negative'],batch_size=100)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(100,100),classes=['positive','negative'],batch_size=100)

images, train_labels = next(train_batches)
vgg16_model = keras.applications.vgg16.VGG16(weights=None,classes=2,input_shape=(100,100,3))
model = Sequential()
for layers in vgg16_model.layers:
    model.add(layers)
model.layers.pop()

for layer in model.layers:
    layer.trainable = False
model.add(Dense(2,activation='softmax'))
model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batches,epochs=20)

test_images, test_labels = next(test_batches)
predict = model.predict_generator(test_batches,steps=7)

test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(128,128),classes=['positive','negative'],batch_size=564)
test_images, test_labels = next(test_batches)
counter = 0
counter2 = 0
for i in range(564):
    if test_labels[i][0] == 0 and (predict[i][0] < predict[i][1]):
        print("test labels before = ",test_labels[i])
        print("test_labels before =",test_labels[i][0]," predict before = ",predict[i][0]," and ",predict[i][1])
        counter += 1
        print("counter before = ",counter)
    elif test_labels[i][0] == 1 and (predict[i][0] > predict[i][1]):
        print("test labels after = ",test_labels[i])
        print("test_labels after =",test_labels[i][0]," predict after = ",predict[i][0]," and ",predict[i][1])
        counter += 1
        print("counter after = ",counter)
    elif test_labels[i][0] == 1 and (predict[i][0] < predict[i][1]):
        print("test labels after = ",test_labels[i])
        print("test_labels after =",test_labels[i][0]," predict after = ",predict[i][0]," and ",predict[i][1])
        counter2 += 1
        print("counter2 after = ",counter2)
    elif test_labels[i][0] == 0 and (predict[i][0] > predict[i][1]):
        print("test labels after = ",test_labels[i])
        print("test_labels after =",test_labels[i][0]," predict after = ",predict[i][0]," and ",predict[i][1])
        counter2 += 1
        print("counter2 after = ",counter2)
print("counter = ",counter)
print("counter2 = ",counter2)
accuracy  =  (counter/564)

