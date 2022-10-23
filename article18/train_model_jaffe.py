import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.applications import EfficientNetB0, DenseNet169, VGG16
from keras.utils.np_utils import to_categorical
from keras import layers, Model, Input
import numpy as np
import pandas as pd

data_path = '../../jaffedbase'
dataset = 'jaffedbase'
data_dir_list = os.listdir(data_path)

img_rows=256
img_cols=256
num_channel=1

num_epoch=50

img_data_list=[]


#for dataset in data_dir_list:
img_list=os.listdir(data_path+'/'+ dataset)
print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
for img in img_list:
    if img != '.DS_Store':
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #print("image: "+ data_path + '/'+ dataset + '/'+ img + "\n")
        input_img_resize=cv2.resize(input_img,(96,96))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

num_classes = 7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:29]=0 #30
labels[30:59]=1 #29
labels[60:92]=2 #32
labels[93:124]=3 #31
labels[125:155]=4 #30
labels[156:187]=5 #31
labels[188:]=6 #30

names = ['angry','disgust','fear','happy','neutral','sad','surprise']

def getLabel(id):
    return ['angry','disgust','fear','happy','neutral','sad','surprise'][id]

# convert class labels to on-hot encoding
y = to_categorical(labels, num_classes)


#Shuffle the dataset
x,y = shuffle(img_data,y, random_state=2)
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

input_shape=img_data[0].shape

def Model_EfficientNetB0(n_classes=7):

    x = Input(shape=input_shape)
    y = EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax"
        ) (x)
    model = Model(inputs=x, outputs=y)

    return model


def Model_DenseNet169(n_classes=7):

    x = Input(shape=input_shape)
    y = DenseNet169(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')(x)
    model = Model(inputs=x, outputs=y)

    return model

def Model_VGG16(n_classes=7):

    x = Input(shape=input_shape)
    y = VGG16(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')(x)
    model = Model(inputs=x, outputs=y)

    return model


EPOCHS = 50
BATCH = 64
LRATE = 1e-4


# Instance model
#create model
efficientNetB0= Sequential()
efficientNetB0.add(Model_EfficientNetB0())
efficientNetB0.summary()
plot_model(efficientNetB0, to_file='model_efficientNetB0_fer_plot.png', show_shapes=True, show_layer_names=True)
#compile model using accuracy to measure model performance
efficientNetB0.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
efficientNetB0_history = efficientNetB0.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_data=(x_test, y_test))



# Instance model
denseNet169 = Model_DenseNet169()
denseNet169.summary()
plot_model(denseNet169, to_file='model_denseNet169_fer_plot.png', show_shapes=True, show_layer_names=True)
#compile model using accuracy to measure model performance
denseNet169.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
denseNet169_history = denseNet169.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_data=(x_test, y_test))

# Instance model
vgg16 = Model_VGG16()
vgg16.summary()
plot_model(vgg16, to_file='model_vgg16_fer_plot.png', show_shapes=True, show_layer_names=True)
#compile model using accuracy to measure model performance
vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
vgg16_history = vgg16.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_data=(x_test, y_test))


def plot_efficientNetB0_loss(history):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(efficientNetB0_history.history['loss'], label='train_loss')
    plt.plot(efficientNetB0_history.history['val_loss'])
    plt.title("Model's training loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_efficientNetB0_fer.png', dpi=300)
    plt.show()


def plot_efficientNetB0_accuracy(history):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(efficientNetB0_history.history['accuracy'])
    plt.plot(efficientNetB0_history.history['val_accuracy'])
    plt.title("Model's training accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_efficientNetB0_fer.png', dpi=300)
    plt.show()


def plot_denseNet169_loss(history):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(denseNet169_history.history['loss'], label='train_loss')
    plt.plot(denseNet169_history.history['val_loss'])
    plt.title("Model's training loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_denseNet169_fer.png', dpi=300)
    plt.show()


def plot_denseNet169_accuracy(history):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(denseNet169_history.history['accuracy'])
    plt.plot(denseNet169_history.history['val_accuracy'])
    plt.title("Model's training accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_denseNet169_fer.png', dpi=300)
    plt.show()

def plot_vgg16_loss(history):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(vgg16_history.history['loss'], label='train_loss')
    plt.plot(vgg16_history.history['val_loss'])
    plt.title("Model's training loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss_vgg16_fer.png', dpi=300)
    plt.show()

def plot_vgg16_accuracy(history):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(vgg16_history.history['accuracy'])
    plt.plot(vgg16_history.history['val_accuracy'])
    plt.title("Model's training accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy_vgg16_fer.png', dpi=300)
    plt.show()
    

# Plot loss:
plot_efficientNetB0_loss(efficientNetB0_history)
plot_denseNet169_loss(denseNet169_history)
plot_vgg16_loss(vgg16_history)

# Plot accuracy:
plot_efficientNetB0_accuracy(efficientNetB0_history)
plot_denseNet169_accuracy(denseNet169_history)
plot_vgg16_accuracy(vgg16_history)



#save model and architecture to single file
efficientNetB0.save("model_efficientNetB0_fer.h5")
denseNet169.save("model_denseNet169_fer.h5")
vgg16.save("model_vgg16_fer.h5")
print("Saved model to disk")

