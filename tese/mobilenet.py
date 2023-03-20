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
path_dataset = "../../main_dataset/"
train_dataset = "../../main_dataset/train"
test_dataset = "../../main_dataset/test"
val_dataset = ".../../main_dataset/val"

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

# Define the number of folds
k = 5

# Get the list of directories in the training dataset
class_directories_train = os.listdir(train_dataset)

# Split each class directory into k parts
class_parts_train = []
for class_dir in class_directories_train:
    class_path = os.path.join(train_dataset, class_dir)
    class_files = os.listdir(class_path)
    random.shuffle(class_files)  # Shuffle the files to ensure randomness
    class_parts_train.append(np.array_split(class_files, k))

# Get the list of directories in the training dataset
class_directories_val = os.listdir(val_dataset)

# Split each class directory into k parts
class_parts_val = []
for class_dir in class_directories_val:
    class_path = os.path.join(val_dataset, class_dir)
    class_files = os.listdir(class_path)
    random.shuffle(class_files)  # Shuffle the files to ensure randomness
    class_parts_val.append(np.array_split(class_files, k))

k_folds = []
for i in range(k):
    train_files = []
    val_files = []
    for j, class_dir_train in enumerate(class_directories_train):
        class_part_train = class_parts_train[j][i]
        train_files += [os.path.join(train_dataset,class_dir_train, filename) for filename in class_part_train]
    for k, class_dir_val in enumerate(class_directories_val):
        class_part_val = class_parts_val[k][i]
        val_files += [os.path.join(val_dataset, class_dir_val, filename) for filename in class_part_val]
    train_labels = [os.path.split(os.path.dirname(file))[1] for file in train_files]
    val_labels = [os.path.split(os.path.dirname(file))[1] for file in val_files]
    train_df = pd.DataFrame({"filename": train_files, "class": train_labels})
    val_df = pd.DataFrame({"filename": val_files, "class": val_labels})
    k_folds.append((train_df, val_df))

# Perform k-fold cross-validation
for fold, (train_df, val_df) in enumerate(k_folds):
    print("Fold ", fold+1)
    #print("Train ", train_df)
    #print("Val ", val_df)
    train_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
        train_df,
        x_col="filename",
        y_col="class",
        target_size=(224, 224),
        batch_size=32,
        shuffle=True)
    val_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
        val_df,
        x_col="filename",
        y_col="class",
        target_size=(224, 224),
        batch_size=32,
        shuffle=False)

    # Train the model for this fold
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=val_generator,
        validation_steps=len(val_generator))

    # Evaluate the model on the validation set for this fold
    scores = model.evaluate(val_generator, steps=len(val_generator))
    print(f"Validation accuracy for fold {fold+1}: {scores[1]*100}%")

# Unfreeze all layers of the model
for layer in base_model.layers:
    layer.trainable = True

# Use a lower learning rate for fine-tuning
opt = SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model on your own dataset
history = model.fit(train_generator, epochs=10, validation_data=val_generator, class_weight=class_weights)

#save the model
model.save('./vgg16.h5')

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
plt.savefig('accuracy_inceptionv3.png')

#clear plot
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#save
plt.savefig('loss_inceptionv3.png')

#clear plot
plt.clf()