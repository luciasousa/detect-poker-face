#train model with CNN architecture
import os
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


#define datapath
datapath = '../../CK_8classes'
data_dir_list = os.listdir(datapath)
labels = sorted(data_dir_list)
img_data_list = []
img_names = []
detector = dlib.get_frontal_face_detector() # type: ignore
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # type: ignore

#read all images into array
for label in labels:
    print(label)
    img_list=os.listdir(datapath+'/'+ label+'/')
    print ('Loaded the images of dataset-'+'{}\n'.format(label))
    for img in img_list:
        input_img=cv2.imread(datapath + '/'+ label + '/'+ img )
        input_img = cv2.resize(input_img, (96, 96))
        gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        shape = []
        bb = []
        dets = detector(input_img, 1)
        _, scores, idx = detector.run(input_img, 1, -1)
        for i, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.left(), d.top(), d.right(), d.bottom(), scores[i]))
            if d is not None and d.top() >= 0 and d.right() <= input_img.shape[1] and d.bottom() <= input_img.shape[0] and d.left() >= 0:
                predicted = predictor(input_img, d)
                shape.append(shape_to_np(predicted))
                (x, y, w, h) = rect_to_bb(d)
                bb.append((x, y, w, h))
                cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #SECOND - process the image, rotate, crop, increase contrast, remove noise
            for j in range(0,len(shape)):
                #Stage 0: Raw Set
                #img_data_list.append(gray_image)
                #cv2.imshow("image", gray_image)
                #print("image: ", img)
                #cv2.waitKey(0)

                #Stage 1: Rotation Correction Set
                rotated_img, landmarks = rotate(gray_image, shape[i])
                #img_data_list.append(rotated_img)

                #Stage 2: Cropped Set
                cropped_face = cropping(rotated_img, landmarks)
                cropped_face = cv2.resize(cropped_face, (96, 96))
                #img_data_list.append(cropped_face)

                #Stage 3: Intensity Normalization Set
                #daniel não tem isto
                #image_norm = cv2.normalize(rotated_img, None, 0, 255, cv2.NORM_MINMAX)
                #img_data_list.append(image_norm)

                #Stage 4: Histogram Equalization Set
                eq_face = hist_eq(cropped_face)
                #img_data_list.append(eq_face)

                #Stage 5: Smoothed Set
                filtered_face = smooth(eq_face)
                img_data_list.append(filtered_face)

                img_names.append(label+'_'+img)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

num_classes = 8
num_of_samples = img_data.shape[0]

print("number of samples: ",num_of_samples)

names = ['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']

def getLabel(id):
    return ['anger','contempt','disgust','fear','happiness','neutral','sadness','surprise'][id]

# convert class labels to on-hot encoding
labels_int = np.ones((num_of_samples,),dtype='int64')

#for images in folder 'emotion' label 1 and folder 'neutral' label 0
#fer2013
for i in range(num_of_samples):
    name = img_names[i]
    label = name.split('_')[0]
    if label == 'anger':
        labels_int[i] = 0
    elif label == 'contempt':
        labels_int[i] = 1
    elif label == 'disgust':
        labels_int[i] = 2
    elif label == 'fear':
        labels_int[i] = 3
    elif label == 'happiness':
        labels_int[i] = 4
    elif label == 'neutral':
        labels_int[i] = 5
    elif label == 'sadness':
        labels_int[i] = 6
    elif label == 'surprise':
        labels_int[i] = 7
    

y = utils.to_categorical(labels_int, num_classes) 
print(img_data.shape)
print(y.shape)

#split the data into train and test and validation
x_train, x_test, y_train, y_test = train_test_split(img_data, y, test_size=0.1,shuffle=True, random_state=2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=2)

IMAGE_SIZE = [96,96]

model = Sequential(  
    [
        Input(shape=(96,96,1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

#plot the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
Image(retina=True, filename='model_plot.png')

#compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

#train the model 
history = model.fit(x_train, y_train, batch_size=4, epochs=50, validation_data=(x_val, y_val))
#save the model
model.save('../../model_lucia_ck_8classes.h5')

#evaluate the model
score = model.evaluate(x_test, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#print train accuracy
score = model.evaluate(x_train, y_train)
print("Train loss:", score[0])
print("Train accuracy:", score[1])

#print validation accuracy
score = model.evaluate(x_val, y_val)
print("Validation loss:", score[0])
print("Validation accuracy:", score[1])


#plot the accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#save
plt.savefig('accuracy_ck_8classes.png')

#clear plot
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#save
plt.savefig('loss_ck_8classes.png')

plt.clf()

#predict the test set and print the classification report and confusion matrix with number of classes 2 (neutral and not neutral) and target names neutral and not neutral 
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test, y_pred, target_names=names))
print(confusion_matrix(y_test, y_pred))