import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet, preprocess_input
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

from keras.callbacks import EarlyStopping


num_classes = 2
path_dataset = "../../main_dataset/"

train_dataset = "../../main_dataset/train"
test_dataset = "../../main_dataset/test"
val_dataset = "../../main_dataset/val"


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    brightness_range=[0.5, 1.5], # add brightness augmentation
)


val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dataset,
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dataset,
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dataset,
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

from sklearn.utils import class_weight

# Get the class labels

# Calculate the class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes) 

# Convert the class weights to a dictionary
class_weights_dict = dict(enumerate(class_weights))

print("class weights: ", class_weights_dict)

# Load pre-trained VGG16 model without the top layers
base_model = MobileNet(weights='imagenet', include_top=False)

# Freeze layers up to the last convolutional block of VGG16
for layer in base_model.layers:
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

early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train_generator, epochs=50, validation_data=val_generator, class_weight=class_weights_dict, callbacks=[early_stop])

#save the model
model.save('../../mobilenet.h5')


#evaluate the model
scores = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {scores[1]*100}%")
print(f"Test loss: {scores[0]*100}%")

scores = model.evaluate(val_generator, steps=len(val_generator))
print(f"Validation accuracy: {scores[1]*100}%")
print(f"Validation loss: {scores[0]*100}%")

scores = model.evaluate(train_generator, steps=len(train_generator))
print(f"Train accuracy: {scores[1]*100}%")
print(f"Train loss: {scores[0]*100}%")

#evaluate the model on the test dataset
model.evaluate(test_generator)

#plot the accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#save
plt.savefig('accuracy_mobilenet.png')

#clear plot
plt.clf()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

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
