#using a pre-trained model InceptionV3 to classify the emotion and using a dataset to test the model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Dropout
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from keras.callbacks import EarlyStopping
import pandas as pd
import os
import shutil
import random
import math

num_classes = 2

path_dataset = "../../main_dataset"
train_dataset = "../../main_dataset/train"
test_dataset = "../../main_dataset/test"
val_dataset = "../../main_dataset/val"

sets = ['train', 'test', 'val']
labels = ['neutral', 'notneutral']



'''
for set in sets:
    for label in labels:
        if label == 'neutral' and set == 'train':
            img_list=os.listdir(path_dataset+'/'+ set + '/' + label + '/')
            for img in img_list:
                input_img=cv2.imread(path_dataset + '/'+ set +'/'+ label + '/'+ img )
                #convert to gray
                input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                #change intensity of image
                input_img_1 = cv2.convertScaleAbs(input_img, alpha=0.6, beta=0.6)
                input_img_2 = cv2.convertScaleAbs(input_img, alpha=1.5, beta=1.5)
                input_img_resize_1=cv2.resize(input_img_1,(96,96))
                input_img_resize_2=cv2.resize(input_img_2,(96,96))
                
                #cv2.imshow('img', input_img_resize)
                #cv2.waitKey(0)
                #save image to folder
                cv2.imwrite(path_dataset + '/'+ set + '/'+ label + '/' + 'da_06' + img, input_img_resize_1)
                cv2.imwrite(path_dataset + '/'+ set + '/'+ label + '/' + 'da_15' + img, input_img_resize_2)

'''


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

# Load the InceptionV3 model without the top layer
inception_model = InceptionV3(weights='imagenet', include_top=False)

# Determine the number of layers in the InceptionV3 model
num_layers = len(inception_model.layers)

#freeze all layers
for layer in inception_model.layers:
    layer.trainable = False

# Add your own top layers to the model
x = GlobalAveragePooling2D()(inception_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  
output = Dense(num_classes, activation='softmax')(x)
inception_model = Model(inputs=inception_model.input, outputs=output)

# Compile the model with a low learning rate
opt = Adam(lr=0.001)
inception_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
inception_model.summary()

# Fine-tune the model on your own dataset
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = inception_model.fit(train_generator, epochs=50, validation_data=val_generator, class_weight=class_weights_dict, callbacks=[early_stop])

#save the model
inception_model.save('../../inceptionv3.h5')

#evaluate the model on the test dataset
scores = inception_model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {scores[1]*100}%")
print(f"Test loss: {scores[0]*100}%")

scores = inception_model.evaluate(val_generator, steps=len(val_generator))
print(f"Validation accuracy: {scores[1]*100}%")
print(f"Validation loss: {scores[0]*100}%")

scores = inception_model.evaluate(train_generator, steps=len(train_generator))
print(f"Train accuracy: {scores[1]*100}%")
print(f"Train loss: {scores[0]*100}%")

#evaluate the model on the test dataset
inception_model.evaluate(test_generator)

#plot the accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#save
plt.savefig('accuracy_inceptionv3.png')

#clear plot
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

#save
plt.savefig('loss_inceptionv3.png')

#clear plot
plt.clf()

#predict the test set and print the classification report and confusion matrix with number of classes 2 (neutral and not neutral) and target names neutral and not neutral 
y_pred = inception_model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=['neutral', 'notneutral']))
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))

