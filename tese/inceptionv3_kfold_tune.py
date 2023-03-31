#using a pre-trained model InceptionV3 to classify the emotion and using a dataset to test the model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import os
import shutil
import random
import math

#To Improve
"""
Increase the number of epochs during training: 
The current code only trains the model for a single epoch, 
which may not be enough for the model to converge to its optimal weights. 
You can try increasing the number of epochs and monitor the validation accuracy to determine if the model is still improving.

Fine-tune more layers of the pre-trained model: 
The current code freezes the first 279 layers of the InceptionV3 model, 
which may not be enough for the model to learn the features necessary for the emotion classification task. 
You can try unfreezing more layers and fine-tuning the model to improve its performance.

Adjust the learning rate: 
The learning rate determines the step size taken during training and can have a significant impact on the training process. 
You can try adjusting the learning rate to find a value that helps the model converge faster and improve its performance.
"""

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
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dataset,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dataset,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

#higher weight for the class with less samples (neutral class)
class_weights = {0: 1., 1: 1.}

# Load the InceptionV3 model without the top layer
inception_model = InceptionV3(weights='imagenet', include_top=False)

# Freeze the first 249 layers of the model - correspondent to the early convolutional layers
for layer in inception_model.layers[:279]:
    layer.trainable = False

for layer in inception_model.layers[279:]:
    layer.trainable = True

# Add your own top layers to the model
x = GlobalAveragePooling2D()(inception_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inception_model.input, outputs=output)

# Compile the model with a low learning rate
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

''' K-FOLD CROSS VALIDATION'''

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
        target_size=(299, 299),
        batch_size=32,
        shuffle=True)
    val_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
        val_df,
        x_col="filename",
        y_col="class",
        target_size=(299, 299),
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

'''END OF K-FOLD CROSS VALIDATION'''

'''FINE TUNING'''

# Unfreeze all layers of the model
for layer in inception_model.layers:
    layer.trainable = True

# Use a lower learning rate for fine-tuning
opt = SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model on your own dataset
history = model.fit(train_generator, epochs=50, validation_data=val_generator, class_weight=class_weights)

#save the model
model.save('../../inceptionv3.h5')

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

#predict the test set and print the classification report and confusion matrix with number of classes 2 (neutral and not neutral) and target names neutral and not neutral 
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report')
cr = classification_report(test_generator.classes, y_pred, target_names=['neutral', 'notneutral'])
print(cr)
cr.plot()
plt.savefig('report_inceptionv3.png')

plt.clf()

print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
print(cm)
cm.plot()
plt.savefig('matrix_inceptionv3.png')

plt.clf()