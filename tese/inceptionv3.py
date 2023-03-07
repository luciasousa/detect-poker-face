#using a pre-trained model InceptionV3 to classify the emotion and using a dataset to test the model
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model
import os

num_classes = 2

#load the model, with the weights pre-trained on ImageNet
inception_model = InceptionV3(weights='imagenet', include_top=False)
# Remove the last layer of the InceptionV3 model
inception_model.layers.pop()

x = GlobalAveragePooling2D()(inception_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# Create a new model with the modified output layers
model = Model(inputs=inception_model.input, outputs=output)

path_dataset = "../../dataset_ferck"

#define classes
classes = ['neutral', 'notneutral']

correct = 0
incorrect = 0
total = 0

# Loop through each image in the dataset
for root, dirs, files in os.walk(path_dataset):
    #classes = dirs
    for file in files:
        # Load and preprocess the image
        image = load_img(os.path.join(root, file), target_size=(299, 299))
        image = img_to_array(image)
        image = preprocess_input(image)

        # Use the pre-trained model to classify the image
        predictions = model.predict(image.reshape(1, 299, 299, 3))
        predicted_class = classes[np.argmax(predictions)]
        # Print the predicted class
        # print(f"Image {file} is a {predicted_class}")
        #check how many images were classified correctly and how many were not
        if predicted_class == root.split('/')[-1]:
            correct += 1
        else:
            incorrect += 1
        total += 1

print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
accuracy = correct / total
print(f"Accuracy: {accuracy}")






