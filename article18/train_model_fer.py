import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import pandas as pd

def load_dataset(net=True):
    # Load and filter in Training/not Training data:
    df = pd.read_csv('../../fer2013/fer2013.csv')
    training = df.loc[df['Usage'] == 'Training']
    testing = df.loc[df['Usage'] != 'Testing']

    # x_train values:
    x_train = np.zeros((training.shape[0], 48, 48))
    for i, pixels in enumerate(training['pixels']):
        x_train[i] = np.array(pixels.split(' ')).reshape((48, 48)).astype('float32')
    

    # x_test values:
    x_test = np.zeros((testing.shape[0], 48, 48))
    for i, pixels in enumerate(testing['pixels']):
        x_test[i] = np.array(pixels.split(' ')).reshape((48, 48)).astype('float32')
        

    # y_train values:
    y_train = training['emotion'].values
    y_train = keras.utils.to_categorical(y_train, 7)

    # y_test values
    y_test = testing['emotion'].values
    y_test = keras.utils.to_categorical(y_test, 7)

    # Reshape data:
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

    return (x_train, y_train), (x_test, y_test)

# load the data
(x_train, y_train) , (x_test, y_test) = load_dataset()

def Model_EfficientNetB0(n_classes=7):

    x = Input(shape=(48, 48, 1))
    y = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x, input_shape=(48, 48, 1), pooling='avg')(x)
    
    # Create model:
    model = Model(x, y)

    # Compile model:
    opt = SGD(lr=LRATE, momentum=0.9, decay=LRATE/EPOCHS)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model

def Model_DenseNet169(n_classes=7):

    x = Input(shape=(48, 48, 1))
    y = DenseNet169(include_top=False, weights='imagenet', input_tensor=x, input_shape=(48, 48, 1), pooling='avg')(x)
    
    # Create model:
    model = Model(x, y)

    # Compile model:
    opt = SGD(lr=LRATE, momentum=0.9, decay=LRATE/EPOCHS)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model

def Model_VGG16(n_classes=7):

    x = Input(shape=(48, 48, 1))
    y = VGG16(include_top=False, weights='imagenet', input_tensor=x, input_shape=(48, 48, 1), pooling='max')(x)
    
    # Create model:
    model = Model(x, y)

    # Compile model:
    opt = SGD(lr=LRATE, momentum=0.9, decay=LRATE/EPOCHS)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model
    
EPOCHS = 50
BATCH = 64
LRATE = 1e-4

# Instance model
efficientNetB0 = Model_EfficientNetB0()
efficientNetB0.summary()
plot_model(efficientNetB0, to_file='model_efficientNetB0_fer_plot.png', show_shapes=True, show_layer_names=True)
#compile model using accuracy to measure model performance
efficientNetB0.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
efficientNetB0_history = efficientNetB0.fit(x_train, y_train,
                   validation_data=(x_test, y_test),
                   epochs=EPOCHS, batch_size=BATCH)

# Instance model
denseNet169 = Model_DenseNet169()
denseNet169.summary()
plot_model(denseNet169, to_file='model_denseNet169_fer_plot.png', show_shapes=True, show_layer_names=True)
#compile model using accuracy to measure model performance
denseNet169.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
denseNet169_history = denseNet169.fit(x_train, y_train,
                   validation_data=(x_test, y_test),
                   epochs=EPOCHS, batch_size=BATCH)

# Instance model
vgg16 = Model_VGG16()
vgg16.summary()
plot_model(vgg16, to_file='model_vgg16_fer_plot.png', show_shapes=True, show_layer_names=True)
#compile model using accuracy to measure model performance
vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
vgg16_history = vgg16.fit(x_train, y_train,
                   validation_data=(x_test, y_test),
                   epochs=EPOCHS, batch_size=BATCH)


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

