import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet, preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import os
import shutil
import random
import math
import cv2


num_classes = 2
path_dataset = "../../../main_dataset/main_dataset/"

train_dataset = "../../../main_dataset/main_dataset/train"
test_dataset = "../../../main_dataset/main_dataset/test"
val_dataset = "../../../main_dataset/main_dataset/val"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10, # randomly rotate images by up to 10 degrees
    zoom_range=0.1, # randomly zoom in by up to 10%
    width_shift_range=0.1, # randomly shift images horizontally by up to 10%
    height_shift_range=0.1, # randomly shift images vertically by up to 10%
    shear_range=0.1, # randomly shear images by up to 10%
    horizontal_flip=True, # randomly flip images horizontally
    fill_mode='nearest' # fill any empty pixels with the nearest valid pixel
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dataset,
    target_size=(299, 299),
    batch_size=1,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dataset,
    target_size=(299, 299),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dataset,
    target_size=(299, 299),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

#higher weight for the class with less samples (neutral class)
class_weights = {0: 1., 1: 1.}

# Load pre-trained VGG16 model without the top layers
base_model = MobileNet(weights='imagenet', include_top=False)

# Freeze layers up to the last convolutional block of VGG16
for layer in base_model.layers[:-22]:
    layer.trainable = False

#add top layers to the base model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
# Create new model with the VGG16 base and the top layers
model = Model(inputs=base_model.input, outputs=x)

# Compile the model with a low learning rate
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Fine-tune the model on your own dataset
history = model.fit(train_generator, epochs=10, validation_data=val_generator, class_weight=class_weights)

#save the model
model.save('./mobilenet.h5')

#evaluate the model on the test dataset
model.evaluate(test_generator)

#plot the accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#save
plt.savefig('accuracy_mobilenet.png')

#clear plot
plt.clf()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#save
plt.savefig('loss_mobilenet.png')

#clear plot
plt.clf()

#predict the test set and print the classification report and confusion matrix with number of classes 2 (neutral and not neutral) and target names neutral and not neutral 
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=['neutral', 'notneutral']))
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
