import os
from cv2 import resize
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import EfficientNetB0
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
base_model = EfficientNetB0(input_shape=IMAGE_SIZE + [3],include_top=False,weights="imagenet")

for layer in base_model.layers[:-4]:
    layer.trainable=False

x = Flatten()(base_model.output)
prediction = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction)

model.summary()

plot_model(model, to_file='efficientNetB0_JAFFE.png', show_shapes=True,show_layer_names=True)
Image(filename='efficientNetB0_JAFFE.png') 

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Training
history = model.fit(x_train, y_train, batch_size=7, epochs=50, verbose=1, validation_data=(x_test, y_test))

#save model
model.save('efficientNetB0_JAFFE.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_efficientNetB0_JAFFE.png')