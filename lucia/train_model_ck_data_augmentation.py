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
import dlib

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2,2))

# Preprocessing block
def hist_eq(img):
	#call the .apply method on the CLAHE object to apply histogram equalization
    return clahe.apply(img)

# Convert the facial landmarks dlib format to numpy
def shape_to_np(shape):
	# pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68 (x, y)-coordinates
	# that map to facial structures on the face
	landmarks = np.zeros((68,2), dtype = int)
	for i in range(0,68):
		landmarks[i] = (shape.part(i).x, shape.part(i).y)
	return landmarks

# take the bounding box predicted by dlib library
# and convert it into (x, y, w, h) where x, y are coordinates
# and w, h are width and height
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

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

# Crop the face
def crop_face(gray_image, shape):
    aux = shape[4] - shape[12]
    distance = np.linalg.norm(aux)
    h = int(distance * 0.1)
    tl = (int((shape[36][0]+shape[18][0])/2), shape[18][1]-h)
    br = (int((shape[45][0]+shape[25][0])/2), int((shape[57][1]+(shape[10][1]+shape[11][1])/2)/2))
    roi = gray_image[tl[1]:br[1],tl[0]:br[0]]
    return roi

detector = dlib.get_frontal_face_detector() #type: ignore
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #type: ignore


#define datapath
datapath = '../../CK_data_augmentation'
data_dir_list = os.listdir(datapath)
labels = sorted(data_dir_list)
img_data_list = []
img_names = []
count_neutral = 0
count_emotion = 0


'''
alpha = 1.5 # Contrast control
beta = 10 # Brightness control

for label in labels:
    if label == 'notneutral':
        img_list=os.listdir(datapath+'/'+ label+'/')
        for img in img_list:
            input_img=cv2.imread(datapath + '/'+ label + '/'+ img )
            #convert to gray
            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            #change intensity of image
            input_img = cv2.convertScaleAbs(input_img, alpha=alpha, beta=beta)
            input_img_resize=cv2.resize(input_img,(96,96))

            #save image to folder
            cv2.imwrite(datapath + '/'+ label + '/'+ img+'_alpha_15_beta_10', input_img_resize)

alpha = 0.5 # Contrast control
beta = 10 # Brightness control

for label in labels:
    if label == 'neutral':
        img_list=os.listdir(datapath+'/'+ label+'/')
        for img in img_list:
            input_img=cv2.imread(datapath + '/'+ label + '/'+ img )
            #convert to gray
            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            #change intensity of image
            input_img = cv2.convertScaleAbs(input_img, alpha=alpha, beta=beta)
            input_img_resize=cv2.resize(input_img,(96,96))
            #save image to folder
            cv2.imwrite(datapath + '/'+ label + '/'+ img +'_alpha_05_beta_10', input_img_resize)

    if label == 'notneutral':
        img_list=os.listdir(datapath+'/'+ label+'/')
        for img in img_list:
            input_img=cv2.imread(datapath + '/'+ label + '/'+ img )
            #convert to gray
            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            #change intensity of image
            input_img = cv2.convertScaleAbs(input_img, alpha=alpha, beta=beta)
            input_img_resize=cv2.resize(input_img,(96,96))

            #save image to folder
            cv2.imwrite(datapath + '/'+ label + '/'+ img+'_alpha_15_beta_10', input_img_resize)
'''
#read all images into array
for label in labels:
    img_list=os.listdir(datapath+'/'+ label+'/')
    print ('Loaded the images of dataset-'+'{}\n'.format(label))
    for img in img_list:
        input_img=cv2.imread(datapath + '/'+ label + '/'+ img )
        input_img = cv2.resize(input_img, (96, 96))
        gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img_data_list.append(gray_image)
        img_names.append(label+'_'+img)
        if label == 'neutral':
            count_neutral += 1
        else:
            count_emotion += 1

print('count_neutral: ', count_neutral)
print('count_emotion: ', count_emotion)
    
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

num_classes = 2
num_of_samples = img_data.shape[0]

names = ['neutral','not neutral']

def getLabel(id):
    return ['neutral','not neutral'][id]

# convert class labels to on-hot encoding
labels_int = np.ones((num_of_samples,),dtype='int64')

#for images in folder 'emotion' label 1 and folder 'neutral' label 0
count_neutral_b = 0
count_emotion_b = 0
for i in range(num_of_samples):
    name = img_names[i]
    label = name.split('_')[0]
    if label == 'neutral':
        labels_int[i] = 0
        count_neutral_b += 1
    else:
        labels_int[i] = 1
        count_emotion_b += 1
    

print('count_neutral_b: ', count_neutral_b)
print('count_emotion_b: ', count_emotion_b)

y = utils.to_categorical(labels_int, num_classes) 
print(img_data.shape)
print(y.shape)

#split the data into train and test and validation
x_train, x_test, y_train, y_test = train_test_split(img_data, y, test_size=0.1,shuffle=True, random_state=2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=2)

# Create a mask for the images to be flipped
flip_mask = np.random.rand(x_train.shape[0]) < 0.5 # type: ignore
# Flip the images in the dataset
x_train[flip_mask] = x_train[flip_mask, ::-1] # type: ignore

flip_count = np.sum(flip_mask)
print(f"Number of images flipped: {flip_count}")

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
model.save('../../model_lucia_ck.h5')

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
plt.savefig('accuracy_ck.png')

#clear plot
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#save
plt.savefig('loss_ck.png')

plt.clf()

#predict the test set and print the classification report and confusion matrix with number of classes 2 (neutral and not neutral) and target names neutral and not neutral 
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test, y_pred, target_names=names))
print(confusion_matrix(y_test, y_pred))