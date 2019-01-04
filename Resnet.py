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
%matplotlib inline
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import *
import numpy as np
from keras.applications import ResNet50

train_path = 'Path_for_Images'
test_path = 'Path_for_Images'

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(197,197),classes=['positive','negative'],batch_size=1)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(197,197),classes=['positive','negative'],batch_size=1)

images, train_labels = next(train_batches)
resnet_model = ResNet50(include_top=True,weights='imagenet')
resnet_model = keras.applications.resnet50.ResNet50(weights=None,classes=2,input_shape=(197,197,3))
resnet_model.summary()

for layer in resnet_model.layers:
    layer.trainable = False
resnet_model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
resnet_model.fit_generator(train_batches,epochs=5)


test_images, test_labels = next(test_batches)
predict = resnet_model.predict_generator(test_batches)

test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(197,197),classes=['positive','negative'],batch_size=564)
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

accuracy  =  (counter/564)


test_labels
rounded_test = []
for i in range(564):
    if (int(test_labels[i][0])) == 0:
        rounded_test.append(0)
    else:
        rounded_test.append(1)

rounded_prediction = []
for i in range(564):
    if predict[i][0] < predict[i][1]:
        rounded_prediction.append(0)
    else:
        rounded_prediction.append(1)

%matplotlib inline 
cm = confusion_matrix(rounded_test,rounded_prediction)

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

    cm_plot_labels = ['negative','positive']
plot_confusion_matrix(cm, cm_plot_labels,title='confusion matrix')
