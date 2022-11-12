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
datapath = '../../ck/CK+48'
data_dir_list = os.listdir(datapath)
neutral_instances = 0
labels = sorted(data_dir_list)
num_classes = 7

img_data_list = []
#array x with all images
for label in labels:
    img_list=os.listdir(datapath+'/'+ label+'/')
    print ('Loaded the images of dataset-'+'{}\n'.format(label))
    for img in img_list:
        input_img=cv2.imread(datapath + '/'+ label + '/'+ img )
        #convert to gray
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(28,28))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

num_classes = 7
num_of_samples = img_data.shape[0]

names = ['anger','contempt','disgust','fear','happy','sadness','surprise']

def getLabel(id):
    return ['anger','contempt','disgust','fear','happy','sadness','surprise'][id]

# convert class labels to on-hot encoding
labels_int = np.ones((num_of_samples,),dtype='int64')
labels_int[0:134]=0 #135
labels_int[135:188]=1 #54
labels_int[189:365]=2 #177
labels_int[366:440]=3 #75
labels_int[441:647]=4 #207
labels_int[648:731]=5 #84
labels_int[732:]=6 #249

y = keras.utils.to_categorical(labels_int, num_classes)
print(img_data.shape)
print(y.shape)
#split the dataset
x_train, x_test, y_train, y_test = train_test_split(img_data, y, test_size=0.1, random_state=2)

IMAGE_SIZE = [28, 28]

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True,show_layer_names=True)
Image(filename='model_plot.png') 

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Training
#shuffle the dataset after each epoch
history = model.fit(x_train, y_train, batch_size=4, epochs=50, verbose=1, validation_data=(x_test, y_test), shuffle=True)

#save model
model.save('model_trained.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')