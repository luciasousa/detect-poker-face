import os
from cv2 import resize
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.utils import plot_model
from IPython.display import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#define datapath
datapath = '../../jaffedbase'
data_dir_list = os.listdir(datapath)

img_rows=256
img_cols=256
num_channel=1

img_data_list=[]
img_names = []

for dataset in data_dir_list:
    img_list=os.listdir(datapath+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(datapath + '/'+ dataset + '/'+ img )
        #split the name of the image
        img_name = img.split('.')[1]
        img_name = ''.join(i for i in img_name if not i.isdigit())
        img_names.append(img_name)
        input_img_resize=cv2.resize(input_img,(96,96))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

#define the number of classes
num_classes = 7
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

for i in range(0, len(img_names)):
    if img_names[i] == 'HA':
        labels[i] = 0
    elif img_names[i] == 'AN':
        labels[i] = 1
    elif img_names[i] == 'FE':
        labels[i] = 2
    elif img_names[i] == 'NE':
        labels[i] = 3
    elif img_names[i] == 'SA':
        labels[i] = 4
    elif img_names[i] == 'SU':
        labels[i] = 5
    elif img_names[i] == 'DI':
        labels[i] = 6

names = ['HAPPY', 'ANGRY', 'FEAR', 'NEUTRAL', 'SAD', 'SURPRISE', 'DISGUST']

def getLabel(id):
    return ['HAPPY', 'ANGRY', 'FEAR', 'NEUTRAL', 'SAD', 'SURPRISE', 'DISGUST'][id]

#show all images and labels
# for i in range(0,213):
#     #print name of image file
#     print('Image file name: ', img_names[i])
#     plt.imshow(img_data[i])
#     plt.title(getLabel(labels[i]))
#     plt.show()

# convert class labels to on-hot encoding
y = keras.utils.to_categorical(labels, num_classes)
# shuffle the dataset
x,y = shuffle(img_data,y, random_state=2)
# split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

IMAGE_SIZE = [96, 96]
base_model = VGG16(input_shape=IMAGE_SIZE + [3],include_top=False,weights="imagenet")

# add a global spatial average pooling layer
x = base_model.output
x = Flatten()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer 
predictions = Dense(7, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
history = model.fit(x_train, y_train, epochs=50, batch_size=4 , validation_data=(x_test, y_test), shuffle=True)

#save model
model.save('models/vgg16_JAFFE.h5')

#print test, train and validation accuracy
#evaluate the model
score = model.evaluate(x_test, y_test, verbose=0, batch_size=4)
#print test accuracy
print('Test accuracy:', score[1])
print('Train accuracy: ', history.history['accuracy'][-1])
print('Validation accuracy: ', history.history['val_accuracy'][-1])

