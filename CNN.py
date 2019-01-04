import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation,MaxPool2D,Dropout
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib
import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot as plt
from PIL import Image

train_path = 'Path'
test_path = 'Path'

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(128,128),classes=['positive','negative'],batch_size=1)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(128,128),classes=['positive','negative'],batch_size=1)

images_train, train_labels = next(train_batches)
print(images_train)
clientImage = np.asarray(images_train, dtype=np.uint8).reshape(128, 128, 3)
plt.imshow(clientImage, interpolation='none');
plt.title(train_labels)

for i in range(3):
    plt.subplot(2,2,i+1)
    print(images_train.shape)
    clientImage = np.asarray(images_train, dtype=np.uint8).reshape(128, 128, 3)
    plt.imshow(clientImage, interpolation='none');
    plt.title(train_labels)
    images_train, train_labels = next(train_batches)

model = Sequential()
model.add(Conv2D(8,kernel_size=(3,3),padding='same',input_shape=(128,128,3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(3,3)))

model.add(Conv2D(16,kernel_size=(3,3),padding='same',input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),padding='same',input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),padding='same',input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))
model.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batches,epochs=5)
test_images, test_labels = next(test_batches)
predict = model.predict_generator(test_batches)

images_test , test_labels  = next(test_batches)
%matplotlib inline  
print(images_test.shape)
clientImage = np.asarray(images_test, dtype=np.uint8).reshape(128, 128, 3)
print(clientImage.shape)
print(test_labels)
plt.imshow(clientImage, interpolation='nearest');

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

%matplotlib inline  
images_test, test_labels = next(test_batches)
clientImage = np.asarray(images_test, dtype=np.uint8).reshape(128, 128, 3)
plt.imshow(clientImage, interpolation='none');
print(test_labels)
print(predict[13])