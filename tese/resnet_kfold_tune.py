import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.applications.resnet import ResNet50, preprocess_input
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

#higher weight for the class with less samples (neutral class)
class_weights = {0: weight_for_0, 1: weight_for_1}

# Load pre-trained resnet model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze layers up to the last convolutional block of resnet
for layer in base_model.layers[:-22]:
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
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=val_generator,
        validation_steps=len(val_generator))
    
    # Evaluate the model on the validation set for this fold
    scores = model.evaluate(val_generator, steps=len(val_generator))
    print(f"Validation accuracy for fold {fold+1}: {scores[1]*100}%")
    print(f"Validation loss for fold {fold+1}: {scores[0]*100}%")

    #train accuracy
    scores = model.evaluate(train_generator, steps=len(train_generator))
    print(f"Train accuracy for fold {fold+1}: {scores[1]*100}%")
    print(f"Train loss for fold {fold+1}: {scores[0]*100}%")

    #graph for the training and validation accuracy
    #plot the accuracy and loss
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #save
    plt.savefig('accuracy_resnet_fold_' + str(fold+1) +'.png')

    #clear plot
    plt.clf()


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    #save
    plt.savefig('loss_resnet_fold_' + str(fold+1) +'.png')

    #clear plot
    plt.clf()


# Unfreeze all layers of the model
for layer in base_model.layers:
    layer.trainable = True

# Use a lower learning rate for fine-tuning
opt = SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model on your own dataset
history = model.fit(train_generator, epochs=50, validation_data=val_generator, class_weight=class_weights)

#save the model
model.save('../../resnet_kfold.h5')

#evaluate the model
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
plt.legend(['train', 'validation'], loc='upper left')
#save
plt.savefig('accuracy_resnet.png')

#clear plot
plt.clf()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

#save
plt.savefig('loss_resnet.png')

#clear plot
plt.clf()

#predict the test set and print the classification report and confusion matrix with number of classes 2 (neutral and not neutral) and target names neutral and not neutral 
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=['neutral', 'notneutral']))
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))

