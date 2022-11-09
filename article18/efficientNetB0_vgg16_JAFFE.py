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

for dataset in data_dir_list:
    img_list=os.listdir(datapath+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(datapath + '/'+ dataset + '/'+ img )
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

labels[0:29]=0 #30
labels[30:59]=1 #29
labels[60:92]=2 #32
labels[93:124]=3 #31
labels[125:155]=4 #30
labels[156:187]=5 #31
labels[188:]=6 #30

names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']

def getLabel(id):
    return ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE'][id]

# convert class labels to on-hot encoding
y = keras.utils.to_categorical(labels, num_classes)

#shuffle the dataset
x,y = shuffle(img_data,y, random_state=2)
#split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

IMAGE_SIZE = [96, 96]
base_model = VGG16(input_shape=IMAGE_SIZE + [3],include_top=False,weights="imagenet")

for layer in base_model.layers[:-4]:
    layer.trainable=False

#load efficientNetB0_JAFFE.h5
efficientNetB0_JAFFE = keras.models.load_model('./models/efficientNetB0_JAFFE.h5')
efficientNetB0_JAFFE._name = 'model3'
#load vgg16_JAFFE.h5
vgg16_JAFFE = keras.models.load_model('./models/vgg16_JAFFE.h5')
vgg16_JAFFE._name = 'model4'

models = [efficientNetB0_JAFFE, vgg16_JAFFE]

model_input = keras.Input(shape=(96, 96, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = layers.Average()(model_outputs)
out_model = keras.Model(inputs=model_input, outputs=ensemble_output)
out_model._name = 'ensemble'
#compile the model
out_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#history
history = out_model.fit(x_train, y_train, batch_size=7, epochs=50, verbose=1, validation_data=(x_test, y_test), shuffle=True)

#save model
out_model.save('models/efficientNetB0_vgg16_JAFFE.h5')

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('accuracy_plots/accuracy_efficientNetB0_vgg16_JAFFE.png')