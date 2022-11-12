import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.utils import plot_model
from IPython.display import Image

train_datagen = ImageDataGenerator( rescale = 1./255,
                                    validation_split = 0.2,                         
                                    rotation_range=5,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale = 1./255,validation_split = 0.2)

test_datagen  = ImageDataGenerator(rescale = 1./255)

resize_and_rescale = keras.Sequential([
  layers.Resizing(96, 96),
  layers.Rescaling(1./255)
])

train_dataset  = train_datagen.flow_from_directory(directory = '../../input/fer2013/train',
                                                   target_size = (96,96),
                                                   class_mode = 'categorical',
                                                   subset = 'training',
                                                   batch_size = 64)
                                    
valid_dataset = valid_datagen.flow_from_directory(directory = '../../input/fer2013/train',
                                                  target_size = (96,96),
                                                  class_mode = 'categorical',
                                                  subset = 'validation',
                                                  batch_size = 64)
                                    
test_dataset = test_datagen.flow_from_directory(directory = '../../input/fer2013/test',
                                                  target_size = (96,96),
                                                  class_mode = 'categorical',
                                                  batch_size = 64)                            

IMAGE_SIZE = [96, 96]
base_model = VGG16(input_shape=IMAGE_SIZE + [3],include_top=False,weights="imagenet")

for layer in base_model.layers[:-4]:
    layer.trainable=False

x = Flatten()(base_model.output)
prediction = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction)

model.summary()

plot_model(model, to_file='models/vgg16_FER.png', show_shapes=True,show_layer_names=True)
Image(filename='models/vgg16_FER.png') 

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_dataset,validation_data=valid_dataset,epochs = 50,verbose = 1, shuffle=True)

#save model 
model.save('models/vgg16_FER.h5')

#evaluate the model
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
#print test accuracy
print('Test accuracy:', test_acc)
print('Train accuracy: ', history.history['accuracy'][-1])
print('Validation accuracy: ', history.history['val_accuracy'][-1])