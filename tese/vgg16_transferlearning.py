import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import os
import shutil
import random
import math
import cv2


num_classes = 2

path_dataset = "../../main_dataset/"
train_dataset = "../../main_dataset/train"
test_dataset = "../../main_dataset/test"
val_dataset = "../../main_dataset/val"

train_datagen = ImageDataGenerator(
    #rotation_range=20,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    zoom_range=0.2,
    #horizontal_flip=True,
    #vertical_flip=True,
    rescale=1./255,
    brightness_range=[0.5, 1.5], # add brightness augmentation
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
neutral = 6821
notneutral = 30209
total = neutral + notneutral

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neutral) * (total / 2.0)
weight_for_1 = (1 / notneutral) * (total / 2.0)

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

class_weights = {0: weight_for_0, 1: weight_for_1}

# Load pre-trained VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False)

# Freeze layers up to the last convolutional block of VGG16
for layer in base_model.layers[:15]:
    layer.trainable = False

#add top layers to the base model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
# Create new model with the VGG16 base and the top layers
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with a low learning rate
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Fine-tune the model on your own dataset
history = model.fit(train_generator, epochs=10, validation_data=val_generator, class_weight=class_weights)

#save the model
model.save('../../vgg16.h5')
#evaluate the model on the test dataset
scores = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {scores[1]*100}%")
scores = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test loss: {scores[0]*100}%")

scores = model.evaluate(val_generator, steps=len(val_generator))
print(f"Validation accuracy: {scores[1]*100}%")
scores = model.evaluate(val_generator, steps=len(val_generator))
print(f"Validation loss: {scores[0]*100}%")

scores = model.evaluate(train_generator, steps=len(train_generator))
print(f"Train accuracy: {scores[1]*100}%")
scores = model.evaluate(train_generator, steps=len(train_generator))
print(f"Train loss: {scores[0]*100}%")

#plot the accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#save
plt.savefig('accuracy_vgg16.png')

#clear plot
plt.clf()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#save
plt.savefig('loss_vgg16.png')

#clear plot
plt.clf()

#predict the test set and print the classification report and confusion matrix with number of classes 2 (neutral and not neutral) and target names neutral and not neutral 
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=['neutral', 'notneutral']))
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
