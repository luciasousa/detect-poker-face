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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import dlib
import tensorflow as tf
from keras.callbacks import EarlyStopping


class IslandLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes):
        super(IslandLoss, self).__init__()
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        # Split the predicted embeddings into class-wise groups
        classwise_embeddings = tf.dynamic_partition(y_pred, tf.argmax(y_true, axis=1), self.num_classes)

        # Calculate the centroids of each class
        centroids = [tf.reduce_mean(embeddings, axis=0, keepdims=True) for embeddings in classwise_embeddings]

        # Calculate the distance between each embedding and its centroid
        distances = [tf.norm(embeddings - centroids[i], axis=1) for i, embeddings in enumerate(classwise_embeddings)]

        # Calculate the intra-class variance and inter-class distance
        intra_class_variance = tf.reduce_sum(distances) / tf.cast(tf.shape(y_pred)[0], tf.float32)
        inter_class_distance = tf.reduce_mean([tf.norm(centroids[i] - centroids[j]) for i in range (self.num_classes) for j in range(i+1, self.num_classes)])

         # Calculate the loss as the sum of intra-class variance and inter-class distance
        loss = intra_class_variance + inter_class_distance

        return loss

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2,2))

# Preprocessing block
def hist_eq(img):
	#call the .apply method on the CLAHE object to apply histogram equalization
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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

img_train_list = []
img_val_list = []
img_test_list = []

img_train = []
img_val = []
img_test = []

labels = ['neutral', 'notneutral']

#for all images in each folder (train, test, val) apply the same preprocessing
#and save the images in arrays
for label in labels:
    img_train_list = os.listdir(train_dataset + "/" + label + "/")
    img_val_list = os.listdir(val_dataset + "/" + label + "/")
    img_test_list = os.listdir(test_dataset + "/" + label + "/")
    for img in img_train_list:
        input_img = cv2.imread(train_dataset + "/" + label + "/" + img)
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
                #img_data_list.append(rotated_img)

                #Stage 2: Cropped Set
                #cropped_face = cropping(rotated_img, landmarks)
                #cropped_face = cv2.resize(cropped_face, (96, 96))
                #img_data_list.append(cropped_face)

                #Stage 3: Intensity Normalization Set
                #daniel não tem isto
                #image_norm = cv2.normalize(rotated_img, None, 0, 255, cv2.NORM_MINMAX)
                #img_data_list.append(image_norm)

                #Stage 4: Histogram Equalization Set
                #eq_face = hist_eq(cropped_face)
                #img_data_list.append(eq_face)

                #Stage 5: Smoothed Set
                #filtered_face = smooth(eq_face)
                #img_data_list.append(filtered_face)

    for img in img_val_list:
        input_img = cv2.imread(val_dataset + "/" + label + "/" + img)
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
                #img_data_list.append(rotated_img)

                #Stage 2: Cropped Set
                #cropped_face = cropping(rotated_img, landmarks)
                #cropped_face = cv2.resize(cropped_face, (96, 96))
                #img_data_list.append(cropped_face)

                #Stage 3: Intensity Normalization Set
                #daniel não tem isto
                #image_norm = cv2.normalize(rotated_img, None, 0, 255, cv2.NORM_MINMAX)
                #img_data_list.append(image_norm)

                #Stage 4: Histogram Equalization Set
                #eq_face = hist_eq(cropped_face)
                #img_data_list.append(eq_face)

                #Stage 5: Smoothed Set
                #filtered_face = smooth(eq_face)
                #img_data_list.append(filtered_face)

    for img in img_test_list:
        input_img = cv2.imread(test_dataset + "/" + label + "/" + img)
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
                #img_data_list.append(rotated_img)

                #Stage 2: Cropped Set
                #cropped_face = cropping(rotated_img, landmarks)
                #cropped_face = cv2.resize(cropped_face, (96, 96))
                #img_data_list.append(cropped_face)

                #Stage 3: Intensity Normalization Set
                #daniel não tem isto
                #image_norm = cv2.normalize(rotated_img, None, 0, 255, cv2.NORM_MINMAX)
                #img_data_list.append(image_norm)

                #Stage 4: Histogram Equalization Set
                #eq_face = hist_eq(cropped_face)
                #img_data_list.append(eq_face)

                #Stage 5: Smoothed Set
                #filtered_face = smooth(eq_face)
                #img_data_list.append(filtered_face)

        
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


'''
model = Sequential(  
    [
        Input(shape=(48,48,1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
'''
'''
# Define the input layer
input_layer = layers.Input(shape=(48,48,1))

# Add three convolutional layers with PReLU activation and batch normalization
conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same')(input_layer)
act1 = layers.PReLU()(conv1)
bn1 = layers.BatchNormalization()(act1)

conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same')(bn1)
act2 = layers.PReLU()(conv2)
bn2 = layers.BatchNormalization()(act2)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)

conv3 = layers.Conv2D(128, kernel_size=(3, 3), activation='linear', padding='same')(pool2)
act3 = layers.PReLU()(conv3)
bn3 = layers.BatchNormalization()(act3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn3)

# Add two fully-connected layers
flatten = layers.Flatten()(pool3)
fc1 = layers.Dense(256, activation='linear')(flatten)
act4 = layers.PReLU()(fc1)
bn4 = layers.BatchNormalization()(act4)
fc2 = layers.Dense(128, activation='linear')(bn4)

# Calculate the island loss at the second fully-connected layer
#il_loss = IslandLoss(2)(fc2) # type: ignore

# Add a softmax layer for classification
softmax = layers.Dense(2, activation='softmax')(fc2)

model = Model(inputs=input_layer, outputs=softmax)
'''
model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])

model.summary()

#early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train_generator, epochs=10, validation_data=val_generator, class_weight=class_weights)

model.save('../../model.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig('accuracy_mymodel_stage0.png')

plt.clf()   # clear figure

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig('loss_mymodel_stage0.png')

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