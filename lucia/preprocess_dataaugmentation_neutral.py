from itertools import chain
import os
import random
from cv2 import resize
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from keras import layers, utils, Input, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.utils import plot_model
from IPython.display import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
import dlib

import random
import shutil

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2,2))

def hist_eq(img):
    return clahe.apply(img)

def shape_to_np(shape):
	landmarks = np.zeros((68,2), dtype = int)
	for i in range(0,68):
		landmarks[i] = (shape.part(i).x, shape.part(i).y)
	return landmarks

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def rotate(gray_image, shape):
	dY = shape[36][1] - shape[45][1]
	dX = shape[36][0] - shape[45][0]
	angle = np.degrees(np.arctan2(dY, dX)) - 180

	rows,cols = gray_image.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
	dst = cv2.warpAffine(gray_image,M,(cols,rows))

	#transform points
	ones = np.ones(shape=(len(shape), 1))
	points_ones = np.hstack([shape, ones])
	new_shape  = M.dot(points_ones.T).T
	new_shape = new_shape.astype(int)

	return dst, new_shape

def smooth(img):
	#Gaussian blurring is highly effective in removing Gaussian noise from an image.
	return cv2.GaussianBlur(img,(3,3),0)

# Crop the face
def cropping(rotated_img, shape):
	aux = shape[4] - shape[12]
	distance = np.linalg.norm(aux)
	h = int(distance * 0.1)
	tl = (int((shape[36][0]+shape[18][0])/2), shape[18][1]-h)
	br = (int((shape[45][0]+shape[25][0])/2), int((shape[57][1]+(shape[10][1]+shape[11][1])/2)/2))
	roi = rotated_img[tl[1]:br[1],tl[0]:br[0]]
	return roi

detector = dlib.get_frontal_face_detector() #type: ignore
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #type: ignore


#define datapath
datapath = '../../main_dataset'
data_dir_list = os.listdir(datapath)
sets = sorted(data_dir_list)
print("list: ", data_dir_list)
labels = ['neutral', 'notneutral']
img_data_list = []
img_names = []
count_neutral = 0
count_emotion = 0

train_dataset = "../../main_dataset/train"
test_dataset = "../../main_dataset/test"
val_dataset = "../../main_dataset/val"


train_neutral = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    brightness_range=[0.5, 1.5], # add brightness augmentation
)
train_emotion= ImageDataGenerator(rescale=1./255)


val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator_neutral= train_neutral.flow_from_directory(directory="../../main_dataset/train/neutral",
                                                     class_mode="categorical",                                                
                                                     batch_size=32)
train_generator_emotion = train_emotion.flow_from_directory(directory="../../main_dataset/train/notneutral",
                                                     class_mode="categorical",                                                
                                                     batch_size=32)


train_generator = chain(train_generator_neutral, train_generator_emotion)

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


img_train_list = []
img_val_list = []
img_test_list = []

img_train = []
img_val = []
img_test = []

labels = ['neutral', 'notneutral']

from sklearn.utils import class_weight

# Get the class labels

# Calculate the class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels) # type: ignore

# Convert the class weights to a dictionary
class_weights_dict = dict(enumerate(class_weights))

print("class weights: ", class_weights_dict)

#for all images in each folder (train, test, val) apply the same preprocessing
#and save the images in arrays
for label in labels:
    img_train_list = os.listdir(train_dataset + "/" + label + "/")
    img_val_list = os.listdir(val_dataset + "/" + label + "/")
    img_test_list = os.listdir(test_dataset + "/" + label + "/")
    for img in img_train_list:
        input_img = cv2.imread(train_dataset+ "/" + label + "/" + img)
        input_img = cv2.resize(input_img, (48, 48))
        gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        shape = []
        bb = []
        dets = detector(input_img, 1)
        _, scores, idx = detector.run(input_img, 1, -1)
        for i, d in enumerate(dets):
            if d is not None and d.top() >= 0 and d.right() <= input_img.shape[1] and d.bottom() <= input_img.shape[0] and d.left() >= 0:
                predicted = predictor(input_img, d)
                shape.append(shape_to_np(predicted))
                (x, y, w, h) = rect_to_bb(d)
                bb.append((x, y, w, h))
                cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for j in range(0,len(shape)):
                #Stage 0: Raw Set
                img_train.append(gray_image)
                #cv2.imshow("image", gray_image)
                #print("image: ", img)
                #cv2.waitKey(0)

                #Stage 1: Rotation Correction Set
                #rotated_img, landmarks = rotate(gray_image, shape[i])
                #img_train.append(rotated_img)

                #Stage 2: Cropped Set
                '''
                if rotated_img.size != 0:
                    cropped_face = cropping(gray_image, landmarks)
                    if cropped_face.size != 0:
                        cropped_face = cv2.resize(cropped_face, (96, 96))
                        img_train.append(cropped_face)
                '''
                #Stage 3: Intensity Normalization Set
                #daniel não tem isto
                #image_norm = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
                #img_train.append(image_norm)

                #Stage 4: Histogram Equalization Set
                #eq_face = hist_eq(gray_image)
                #img_train.append(eq_face)

                #Stage 5: Smoothed Set
                #filtered_face = smooth(gray_image)
                #img_train.append(filtered_face)

    for img in img_val_list:
        input_img = cv2.imread(val_dataset  + "/" + label + "/" + img)
        input_img = cv2.resize(input_img, (48, 48))
        gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        shape = []
        bb = []
        dets = detector(input_img, 1)
        _, scores, idx = detector.run(input_img, 1, -1)
        for i, d in enumerate(dets):
            if d is not None and d.top() >= 0 and d.right() <= input_img.shape[1] and d.bottom() <= input_img.shape[0] and d.left() >= 0:
                predicted = predictor(input_img, d)
                shape.append(shape_to_np(predicted))
                (x, y, w, h) = rect_to_bb(d)
                bb.append((x, y, w, h))
                cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for j in range(0,len(shape)):
                #Stage 0: Raw Set
                img_val.append(gray_image)
                #cv2.imshow("image", gray_image)
                #print("image: ", img)
                #cv2.waitKey(0)

                #Stage 1: Rotation Correction Set
                #rotated_img, landmarks = rotate(gray_image, shape[i])
                #img_val.append(rotated_img)

                #Stage 2: Cropped Set
                '''
                if rotated_img.size != 0:
                    cropped_face = cropping(rotated_img, landmarks)
                    if cropped_face.size != 0:
                        cropped_face = cv2.resize(cropped_face, (96, 96))
                        img_val.append(cropped_face)
                '''

                #Stage 3: Intensity Normalization Set
                #daniel não tem isto
                #image_norm = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
                #img_val.append(image_norm)

                #Stage 4: Histogram Equalization Set
                #eq_face = hist_eq(gray_image)
                #img_val.append(eq_face)

                #Stage 5: Smoothed Set
                #filtered_face = smooth(gray_image)
                #img_val.append(filtered_face)

    for img in img_test_list:
        input_img = cv2.imread(test_dataset  + "/" + label + "/" + img)
        input_img = cv2.resize(input_img, (48, 48))
        gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        shape = []
        bb = []
        dets = detector(input_img, 1)
        _, scores, idx = detector.run(input_img, 1, -1)
        for i, d in enumerate(dets):
            if d is not None and d.top() >= 0 and d.right() <= input_img.shape[1] and d.bottom() <= input_img.shape[0] and d.left() >= 0:
                predicted = predictor(input_img, d)
                shape.append(shape_to_np(predicted))
                (x, y, w, h) = rect_to_bb(d)
                bb.append((x, y, w, h))
                cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for j in range(0,len(shape)):
                #Stage 0: Raw Set
                img_test.append(gray_image)
                #cv2.imshow("image", gray_image)
                #print("image: ", img)
                #cv2.waitKey(0)

                #Stage 1: Rotation Correction Set
                #rotated_img, landmarks = rotate(gray_image, shape[i])
                #img_test.append(rotated_img)

                #Stage 2: Cropped Set
                '''
                if rotated_img.size != 0:
                    cropped_face = cropping(rotated_img, landmarks)
                    if cropped_face.size != 0:
                        cropped_face = cv2.resize(cropped_face, (96, 96))
                        img_test.append(cropped_face)
                '''

                #Stage 3: Intensity Normalization Set
                #daniel não tem isto
                #image_norm = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
                #img_test.append(image_norm)

                #Stage 4: Histogram Equalization Set
                #eq_face = hist_eq(gray_image)
                #img_test.append(eq_face)

                #Stage 5: Smoothed Set
                #filtered_face = smooth(gray_image)
                #img_test.append(filtered_face)

        
img_t= np.array(img_train)
img_t = img_t.astype('float32')
img_t = img_t/255
img_t.shape

        
img_v = np.array(img_val)
img_v = img_v.astype('float32')
img_v = img_v/255
img_v.shape
    
        
img_te = np.array(img_test)
img_te = img_te.astype('float32')
img_te = img_te/255
img_te.shape


#higher weight for the class with less samples (neutral class)
#class_weights = {0: weight_for_0, 1: weight_for_1}

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(299, 299, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


#compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

#train the model
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=[early_stop], class_weight=class_weights_dict)
#save the model
model.save('../../model_preprocess_da.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig('acc_cnn_da_online.png')

plt.clf()   # clear figure

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig('loss_cnn_da_online.png')

plt.clf()   # clear figure

# Evaluate the model on the validation set for this fold
scores = model.evaluate(val_generator, steps=len(val_generator))
print(f"Validation accuracy: {scores[1]*100}%")
print(f"Validation loss: {scores[0]*100}%")

#train accuracy
scores = model.evaluate(train_generator, steps=len(train_generator))
print(f"Train accuracy: {scores[1]*100}%")
print(f"Train loss: {scores[0]*100}%")

scores = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {scores[1]*100}%")
print(f"Test loss: {scores[0]*100}%")

#predict the test set and print the classification report and confusion matrix with number of classes 2 (neutral and not neutral) and target names neutral and not neutral 
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=['neutral', 'notneutral']))
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))