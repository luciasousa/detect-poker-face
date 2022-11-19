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
datapath = '../../jaffedbase'
#datapath = '../../fer_binary/fer'
data_dir_list = os.listdir(datapath)
labels = sorted(data_dir_list)
img_data_list = []
img_names = []



#read all images into array
for label in labels:
    img_list=os.listdir(datapath+'/'+ label+'/')
    print ('Loaded the images of dataset-'+'{}\n'.format(label))
    for img in img_list:
        input_img=cv2.imread(datapath + '/'+ label + '/'+ img )
        #convert to gray
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(48,48))
        img_data_list.append(input_img_resize)
        img_names.append(img)

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

#split the name of the image to get the label and check if equal to 'NE'

for i in range(num_of_samples):
    name = img_names[i]
    label = name.split('.')[1]
    result = ''.join([i for i in label if not i.isdigit()])
    if result == 'NE':
        labels_int[i] = 0
    else:
        labels_int[i] = 1


#for images in folder 'emotion' label 1 and folder 'neutral' label 0
'''
for label in labels:
    img_list=os.listdir(datapath+'/'+ label+'/')
    for i in range(len(img_list)):
        if label == 'NEUTRAL':
            labels_int[i] = 0
        else:
            labels_int[i] = 1
'''
y = keras.utils.to_categorical(labels_int, num_classes)
print(img_data.shape)
print(y.shape)

#split the data into train and test and validation
x_train, x_test, y_train, y_test = train_test_split(img_data, y, test_size=0.2,shuffle=True, random_state=8)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=8)


IMAGE_SIZE = [48,48]

model = keras.Sequential(
    [
        keras.Input(shape=(48,48,1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

for layer in model.layers:
    layer.trainable = True

model.layers[2].trainable = False
model.layers[4].trainable = False

#plot the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
Image(retina=True, filename='model_plot.png')

#compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

#train the model
model.fit(x_train, y_train, batch_size=4, epochs=50, validation_split=0.1)

#save the model
model.save('../../model_lucia.h5')

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


#plot the names of the images with the predicted label and the actual label
for i in range(len(x_test)):
    img = x_test[i]
    img = img.reshape(1,48,48,1)
    prediction = model.predict(img)
    prediction = prediction.argmax()
    print('predicted: ', getLabel(prediction), 'actual: ', getLabel(y_test[i].argmax()))
    #plot the image
    plt.imshow(x_test[i].reshape(48,48), cmap='gray')
    plt.show()
    #wait for key press
    cv2.waitKey(0)

