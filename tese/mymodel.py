import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from keras import layers, utils, Input, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import plot_model
from IPython.display import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import dlib


clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2,2))

# Preprocessing block
def hist_eq(img):
	#call the .apply method on the CLAHE object to apply histogram equalization
    return clahe.apply(img)

def smooth(img):
	#Gaussian blurring is highly effective in removing Gaussian noise from an image.
	return cv2.GaussianBlur(img,(3,3),0)

def resize(img):
	return cv2.resize(img, (32,32))

def to_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	
	return (x, y, w, h)

# Convert the facial landmarks dlib format to numpy
def shape_to_np(shape):
	# pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68 (x, y)-coordinates
	# that map to facial structures on the face
	landmarks = np.zeros((68,2), dtype = int)

	for i in range(0,68):
		landmarks[i] = (shape.part(i).x, shape.part(i).y)

	return landmarks


# Face detection
def detect_face(image, gray_image):
	rects = detector(gray_image, 1)
	shape = []
	bb = []

	for (z, rect) in enumerate(rects):
		if rect is not None and rect.top() >= 0 and rect.right() <= gray_image.shape[1] and rect.bottom() <= gray_image.shape[0] and rect.left() >= 0:
			predicted = predictor(gray_image, rect)
			shape.append(shape_to_np(predicted))
			(x, y, w, h) = rect_to_bb(rect)
			bb.append((x, y, w, h))
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		#for (x, y) in shape:
		#	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	return shape, bb

# Rotation correction
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

# Crop the face by using the facial landmarks of dlib
def cropping(rotated_img, shape):
	aux = shape[4] - shape[12]
	distance = np.linalg.norm(aux)
	h = int(distance * 0.1)
	tl = (int((shape[36][0]+shape[18][0])/2), shape[18][1]-h)
	br = (int((shape[45][0]+shape[25][0])/2), int((shape[57][1]+(shape[10][1]+shape[11][1])/2)/2))
	roi = rotated_img[tl[1]:br[1],tl[0]:br[0]]
	return roi

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

#define classes and print each class and number of samples
classes = train_generator.class_indices
print(classes)

#count number of samples in each class

"""

            Neutro (18%)        Não Neutro (82%)

Teste       1367                6110                7477 (20%)
Treino      4909                21690               26599 (72%)
Validação   545                 2409                2954 (8%)
                                                    37030 (100%)

"""

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
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit_generator(train_generator, epochs=50, validation_data=val_generator, validation_steps=50, class_weight=class_weights)

model.save('../../model.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.clf()   # clear figure

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.clf()   # clear figure

#predict the test set and print the classification report and confusion matrix with number of classes 2 (neutral and not neutral) and target names neutral and not neutral 
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=['neutral', 'notneutral']))
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))