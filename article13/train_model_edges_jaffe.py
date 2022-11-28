import os
from cv2 import resize
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.utils import plot_model
from IPython.display import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#define datapath
datapath = '../../jaffe_7classes'
data_dir_list = os.listdir(datapath)
labels = sorted(data_dir_list)
img_data_list = []
img_names = []
read_images = []


#read all images into array
for label in labels:
    img_list=os.listdir(datapath+'/'+ label+'/')
    print ('Loaded the images of dataset-'+'{}\n'.format(label))
    for img in img_list:
        input_img=cv2.imread(datapath + '/'+ label + '/'+ img )
        read_images.append(input_img)
        #convert to gray
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(48,48))
        img_data_list.append(input_img_resize)
        img_names.append(img)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

num_classes = 7
num_of_samples = img_data.shape[0]

names = ['ANGRY','DISGUST','FEAR','HAPPY','SAD','SURPRISE','NEUTRAL']

def getLabel(id):
    return ['ANGRY', 'DISGUST','FEAR','HAPPY','SAD','SURPRISE','NEUTRAL'][id]

# convert class labels to on-hot encoding
labels_int = np.ones((num_of_samples,),dtype='int64')

#for images in folder 'emotion' label 1 and folder 'neutral' label 0
for label in labels:
    img_list=os.listdir(datapath+'/'+ label+'/')
    for i in range(len(img_list)):
        if label == 'NEUTRAL':
            labels_int[i] = 0
        else:
            labels_int[i] = 1
    
y = keras.utils.to_categorical(labels_int, num_classes)
print(img_data.shape)
print(y.shape)

#convert to canny edges
#convert input image to cannny edge
def canny_edge(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 100, 200)
    return img

img_data_edges = []
for i in range(len(read_images)):
    img_data_edges.append(canny_edge(read_images[i]))
img_data_edges = np.array(img_data_edges)
img_data_edges = img_data_edges.astype('float32')
img_data_edges = img_data_edges/255
img_data_edges.shape

#split the data into train and test and validation
x_train, x_test, y_train, y_test = train_test_split(img_data, y, test_size=0.2,shuffle=True, random_state=8)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle=True, random_state=8)


IMAGE_SIZE = [48,48]

model = keras.Sequential(
    [
        keras.Input(shape=(48,48,1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

#set all layers to not trainable
for layer in model.layers:
    layer.trainable = True
#set layers max pool to not trainable
model.layers[2].trainable = False
model.layers[4].trainable = False
model.layers[6].trainable = False


#compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

#train the model
history = model.fit(x_train, y_train, batch_size=4, epochs=150, validation_split=0.1)

#save the model
model.save('../../model_edges_jaffe.h5')

#evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#print train accuracy
score = model.evaluate(x_train, y_train, verbose=0)
print("Train loss:", score[0])
print("Train accuracy:", score[1])

#print validation accuracy
score = model.evaluate(x_val, y_val, verbose=0)
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
plt.savefig('accuracy_edges_jaffe.png')

#clear plot
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#save
plt.savefig('loss_edges_jaffe.png')

#Test loss: 2.411907189525664e-07
#Test accuracy: 1.0
#Train loss: 1.5300076938729035e-07
#Train accuracy: 1.0
#Validation loss: 1.6356617038582044e-07
#Validation accuracy: 1.0

