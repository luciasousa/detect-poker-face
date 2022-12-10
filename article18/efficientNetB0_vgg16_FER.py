import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, EfficientNetB0
from keras.layers import Dense, Flatten, Input
from keras.models import Model, Sequential
from keras.utils import plot_model
from IPython.display import Image
import numpy as np

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

resize_and_rescale = Sequential([
  layers.Resizing(96, 96),
  layers.Rescaling(1./255)
])

train_dataset  = train_datagen.flow_from_directory(directory = '../../FER2013/train',
                                                   target_size = (96,96),
                                                   class_mode = 'categorical',
                                                   subset = 'training',
                                                   batch_size = 64)
                                    
valid_dataset = valid_datagen.flow_from_directory(directory = '../../FER2013/train',
                                                  target_size = (96,96),
                                                  class_mode = 'categorical',
                                                  subset = 'validation',
                                                  batch_size = 64)
                                    
test_dataset = test_datagen.flow_from_directory(directory = '../../FER2013/test',
                                                  target_size = (96,96),
                                                  class_mode = 'categorical',
                                                  batch_size = 64)                            

#load efficientNetB0_FER.h5
efficientNetB0_FER = models.load_model('./models/efficientNetB0_FER.h5')
efficientNetB0_FER._name = 'model1'
#load vgg16_FER.h5
vgg16_FER = models.load_model('./models/vgg16_FER.h5')
vgg16_FER._name = 'model2'

models = [efficientNetB0_FER, vgg16_FER]

model_input = Input(shape=(96, 96, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = layers.Average()(model_outputs)
ensemble_model = Model(inputs=model_input, outputs=ensemble_output)

#compile the model
ensemble_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#history
history = ensemble_model.fit(train_dataset, epochs=50, validation_data=valid_dataset, shuffle=True)

#save model
ensemble_model.save('models/efficientNetB0_vgg16_FER.h5')

#evaluate the model
test_loss, test_acc = ensemble_model.evaluate(test_dataset, verbose=2)
#print test accuracy
print('Test accuracy:', test_acc)
print('Train accuracy: ', history.history['accuracy'][-1])
print('Validation accuracy: ', history.history['val_accuracy'][-1])