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
import numpy as np
import pandas as pd

def load_dataset(net=True):
    """Utility function to load the FER2013 dataset.
    
    It returns the formated tuples (x_train, y_train) , (x_test, y_test).

    Parameters
    ==========
    net : boolean
        This parameter is used to reshape the data from images in 
        (cols, rows, channels) format. In case that it is False, a standard
        format (cols, rows) is used.
    """

    # Load and filter in Training/not Training data:
    df = pd.read_csv('../../fer2013/fer2013.csv')
    training = df.loc[df['Usage'] == 'Training']
    testing = df.loc[df['Usage'] != 'Testing']

    # x_train values:
    x_train = training[['pixels']].values
    x_train = [np.fromstring(e[0], dtype=int, sep=' ') for e in x_train]
    if net:
        x_train = [e.reshape((48, 48, 1)).astype('float32') for e in x_train]
    else:
        x_train = [e.reshape((48, 48)) for e in x_train]
    x_train = np.array(x_train)

    # x_test values:
    x_test = testing[['pixels']].values
    x_test = [np.fromstring(e[0], dtype=int, sep=' ') for e in x_test]
    if net:
        x_test = [e.reshape((48, 48, 1)).astype('float32') for e in x_test]
    else:
        x_test = [e.reshape((48, 48)) for e in x_test]
    x_test = np.array(x_test)

    # y_train values:
    y_train = training[['emotion']].values
    y_train = keras.utils.to_categorical(y_train)

    # y_test values
    y_test = testing[['emotion']].values
    y_test = keras.utils.to_categorical(y_test)

    return (x_train, y_train) , (x_test, y_test)

# load the data
(x_train, y_train) , (x_test, y_test) = load_dataset()

def ResidualBlock(prev_layer):
    """Residual block from the EDNN model for FER by Deepak Kumar Jaina,
    Pourya Shamsolmoalib & Paramjit Sehdev, as it appears in "Extended 
    deep neural network for facial emotion recognition", 2019.
    """
    conv_1 = Conv2D(64, (1, 1))(prev_layer)
    conv_2 = Conv2D(64, (3, 3), padding="same")(conv_1)
    shortc = concatenate([conv_1, conv_2], axis=-1)
    conv_3 = Conv2D(128, (3, 3), padding="same")(shortc)
    conv_4 = Conv2D(256, (1, 1))(conv_3)
    output = concatenate([conv_4, prev_layer], axis=-1)
    
    return output


def EDNN(n_classes=7):
    """
    EDNN model for FER by Deepak Kumar Jaina, Pourya Shamsolmoalib &
    Paramjit Sehdev, as it appears in "Extended deep neural network for 
    facial emotion recognition", 2019.
    """

    x = Input(shape=(48, 48, 1))
    y = Conv2D(32, (5, 5), input_shape=(48, 48, 1), strides=(2, 2), 
               data_format='channels_last')(x)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Conv2D(64, (3, 3), strides=(1, 1))(y)
    y = ResidualBlock(y)
    y = Conv2D(128, (3, 3), strides=(1, 1), padding="same")(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Conv2D(128, (3, 3), strides=(1, 1))(y)
    y = ResidualBlock(y)
    y = Conv2D(256, (3, 3), strides=(1, 1), padding="same")(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Conv2D(512, (3, 3), strides=(1, 1), padding="same")(y)
    y = Flatten()(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(512, activation='relu')(y)
    y = Dropout(0.2)(y)
    y = Dense(n_classes, activation='softmax')(y)
    
    # Create model:
    model = Model(x, y)

    # Compile model:
    opt = SGD(lr=LRATE, momentum=0.9, decay=LRATE/EPOCHS)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model
    
EPOCHS = 5
BATCH = 64
LRATE = 1e-4

# Instance model
ednn = EDNN()
ednn.summary()
plot_model(ednn, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#compile model using accuracy to measure model performance
ednn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
history = ednn.fit(x_train, y_train,
                   validation_data=(x_test, y_test),
                   epochs=EPOCHS, batch_size=BATCH)

def plot_loss(history):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model's training loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png', dpi=300)
    plt.show()


def plot_accuracy(history):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model's training accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy.png', dpi=300)
    plt.show()
    

# Plot loss:
plot_loss(history)

# Plot accuracy:
plot_accuracy(history)

#save model and architecture to single file
ednn.save("model.h5")
print("Saved model to disk")

