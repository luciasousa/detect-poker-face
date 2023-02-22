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

#define datapath
datapath = '../../FER2013_binary'
data_dir_list = os.listdir(datapath)
labels = sorted(data_dir_list)
img_data_list = []
img_names = []
detector = dlib.get_frontal_face_detector() # type: ignore
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # type: ignore

img_data_list1 = []
accuracy_stage1 = []
loss_stage1 = []
img_data_list2 = []
accuracy_stage2 = []
loss_stage2 = []
img_data_list3 = []
accuracy_stage3 = []
loss_stage3 = []
img_data_list4 = []
accuracy_stage4 = []
loss_stage4 = []
img_data_list5 = []
accuracy_stage5 = []
loss_stage5 = []

for label in labels:
    print(label)
    img_list=os.listdir(datapath+'/'+ label+'/')
    print ('Loaded the images of dataset-'+'{}\n'.format(label))
    for img in img_list:
        input_img=cv2.imread(datapath + '/'+ label + '/'+ img )
        input_img = cv2.resize(input_img, (48, 48))
        gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img_names.append(label+'_'+img)
        shape = []
        bb = []
        dets = detector(input_img, 1)
        _, scores, idx = detector.run(input_img, 1, -1)
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.left(), d.top(), d.right(), d.bottom(), scores[i]))
            if d is not None and d.top() >= 0 and d.right() <= input_img.shape[1] and d.bottom() <= input_img.shape[0] and d.left() >= 0:
                predicted = predictor(input_img, d)
                shape.append(shape_to_np(predicted))
                (x, y, w, h) = rect_to_bb(d)
                bb.append((x, y, w, h))
                cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #SECOND - process the image, rotate, crop, increase contrast, remove noise
            for j in range(0,len(shape)):
                #Stage 0: Raw Set
                img_data_list1.append(gray_image)
                #cv2.imshow("image", gray_image)
                #print("image: ", img)
                #cv2.waitKey(0)

                #Stage 1: Rotation Correction Set
                rotated_img, landmarks = rotate(gray_image, shape[i])
                img_data_list2.append(rotated_img)

                #Stage 2: Cropped Set
                #cropped_face = cropping(rotated_img, landmarks)
                #not the cropped face with shape 0
                #cropped_face = cv2.resize(cropped_face, (48, 48))
                #cropped_face to correct type 
                #img_data_list3.append(cropped_face)

                #Stage 3: Intensity Normalization Set
                #daniel não tem isto
                image_norm = cv2.normalize(rotated_img, None, 0, 255, cv2.NORM_MINMAX)
                img_data_list3.append(image_norm)

                #Stage 4: Histogram Equalization Set
                eq_face = hist_eq(image_norm)
                img_data_list4.append(eq_face)

                #Stage 5: Smoothed Set
                filtered_face = smooth(eq_face)
                img_data_list5.append(filtered_face)

                img_names.append(label+'_'+img)
		
img_data1 = np.array(img_data_list1)
img_data1 = img_data1.astype('float32')
img_data1 = img_data1/255
img_data1.shape

img_data2 = np.array(img_data_list2)
img_data2 = img_data2.astype('float32')
img_data2 = img_data2/255
img_data2.shape

img_data3 = np.array(img_data_list3)
img_data3 = img_data3.astype('float32')
img_data3 = img_data3/255
img_data3.shape

img_data4 = np.array(img_data_list4)
img_data4 = img_data4.astype('float32')
img_data4 = img_data4/255
img_data4.shape

img_data5 = np.array(img_data_list5)
img_data5 = img_data5.astype('float32')
img_data5 = img_data5/255
img_data5.shape

num_classes = 2
num_of_samples = img_data1.shape[0]

print("number of samples: ",num_of_samples)

names = ['neutral','not neutral']

def getLabel(id):
    return ['neutral','not neutral'][id]

# convert class labels to on-hot encoding
labels_int = np.ones((num_of_samples,),dtype='int64')

#for images in folder 'emotion' label 1 and folder 'neutral' label 0
#fer2013
for i in range(num_of_samples):
    name = img_names[i]
    label = name.split('_')[0]
    if label == 'neutral':
        labels_int[i] = 0
    else:
        labels_int[i] = 1
    

y = utils.to_categorical(labels_int, num_classes) 
print(img_data1.shape)
print(img_data2.shape)
print(img_data3.shape)
print(img_data4.shape)
print(img_data5.shape)
print(y.shape)

#split the data into train and test and validation
x_train1, x_test1, y_train1, y_test1 = train_test_split(img_data1, y, test_size=0.2,shuffle=True, random_state=2)
x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train1, y_train1, test_size=0.1, shuffle=True, random_state=2)


x_train2, x_test2, y_train2, y_test2 = train_test_split(img_data2, y, test_size=0.2,shuffle=True, random_state=2)
x_train2, x_val2, y_train2, y_val2 = train_test_split(x_train2, y_train2, test_size=0.1, shuffle=True, random_state=2)

x_train3, x_test3, y_train3, y_test3 = train_test_split(img_data3, y, test_size=0.2,shuffle=True, random_state=2)
x_train3, x_val3, y_train3, y_val3 = train_test_split(x_train3, y_train3, test_size=0.1, shuffle=True, random_state=2)

x_train4, x_test4, y_train4, y_test4 = train_test_split(img_data4, y, test_size=0.2,shuffle=True, random_state=2)
x_train4, x_val4, y_train4, y_val4 = train_test_split(x_train4, y_train4, test_size=0.1, shuffle=True, random_state=2)

x_train5, x_test5, y_train5, y_test5 = train_test_split(img_data5, y, test_size=0.2,shuffle=True, random_state=2)
x_train5, x_val5, y_train5, y_val5 = train_test_split(x_train5, y_train5, test_size=0.1, shuffle=True, random_state=2)

IMAGE_SIZE = [48,48]

model = Sequential(  
    [
        Input(shape=(48,48,1)),
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
history1 = model.fit(x_train1, y_train1, batch_size=4, epochs=50, validation_data=(x_val1, y_val1))
#evaluate the model
score = model.evaluate(x_test1, y_test1)
print("Test loss stage 1:", score[0])
print("Test accuracy stage 1:", score[1])

#print train accuracy
score = model.evaluate(x_train1, y_train1)
print("Train loss stage 1:", score[0])
print("Train accuracy stage 1:", score[1])

#print validation accuracy
score = model.evaluate(x_val1, y_val1)
print("Validation loss stage 1:", score[0])
print("Validation accuracy stage 1:", score[1])

history2 = model.fit(x_train2, y_train2, batch_size=4, epochs=50, validation_data=(x_val2, y_val2))
score = model.evaluate(x_test2, y_test2)
print("Test loss stage 2:", score[0])
print("Test accuracy stage 2:", score[1])

#print train accuracy
score = model.evaluate(x_train2, y_train2)
print("Train loss stage 2:", score[0])
print("Train accuracy stage 2:", score[1])

#print validation accuracy
score = model.evaluate(x_val2, y_val2)
print("Validation loss stage 2:", score[0])
print("Validation accuracy stage 2:", score[1])

history3 = model.fit(x_train3, y_train3, batch_size=4, epochs=50, validation_data=(x_val3, y_val3))
score = model.evaluate(x_test3, y_test3)
print("Test loss stage 3:", score[0])
print("Test accuracy stage 3:", score[1])

#print train accuracy
score = model.evaluate(x_train3, y_train3)
print("Train loss stage 3:", score[0])
print("Train accuracy stage 3:", score[1])

#print validation accuracy
score = model.evaluate(x_val3, y_val3)
print("Validation loss stage 3:", score[0])
print("Validation accuracy stage 3:", score[1])

history4 = model.fit(x_train4, y_train4, batch_size=4, epochs=50, validation_data=(x_val4, y_val4))
score = model.evaluate(x_test4, y_test4)
print("Test loss stage 4:", score[0])
print("Test accuracy stage 4:", score[1])

#print train accuracy
score = model.evaluate(x_train4, y_train4)
print("Train loss stage 4:", score[0])
print("Train accuracy stage 4:", score[1])

#print validation accuracy
score = model.evaluate(x_val4, y_val4)
print("Validation loss stage 4:", score[0])
print("Validation accuracy stage 4:", score[1])

history5 = model.fit(x_train5, y_train5, batch_size=4, epochs=50, validation_data=(x_val5, y_val5))
score = model.evaluate(x_test5, y_test5)
print("Test loss stage 5:", score[0])
print("Test accuracy stage 5:", score[1])

#print train accuracy
score = model.evaluate(x_train5, y_train5)
print("Train loss stage 5:", score[0])
print("Train accuracy stage 5:", score[1])

#print validation accuracy
score = model.evaluate(x_val5, y_val5)
print("Validation loss stage 5:", score[0])
print("Validation accuracy stage 5:", score[1])

#plot the accuracy for all stages in one plot 
plt.plot(history1.history['accuracy'])
plt.plot(history2.history['accuracy'])
plt.plot(history3.history['accuracy'])
plt.plot(history4.history['accuracy'])
plt.plot(history5.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['phase 0', 'phase 1', 'phase 2', 'phase 3', 'phase 4'], loc='upper left')
plt.show()

#plot the loss for all stages in one plot
plt.plot(history1.history['loss'])
plt.plot(history2.history['loss'])
plt.plot(history3.history['loss'])
plt.plot(history4.history['loss'])
plt.plot(history5.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['phase 0', 'phase 1', 'phase 2', 'phase 3', 'phase 4'], loc='upper left')
plt.show()
