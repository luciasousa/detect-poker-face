import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, EfficientNetB0
from keras.layers import Dense, Flatten
from keras.models import Model
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

#load efficientNetB0_FER.h5
efficientNetB0_FER = keras.models.load_model('./models/efficientNetB0_FER.h5')
efficientNetB0_FER._name = 'model1'
#load vgg16_FER.h5
vgg16_FER = keras.models.load_model('./models/vgg16_FER.h5')
vgg16_FER._name = 'model2'

models = [efficientNetB0_FER, vgg16_FER]

model_input = keras.Input(shape=(96, 96, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = layers.Average()(model_outputs)
ensemble_model = keras.Model(inputs=model_input, outputs=ensemble_output)

#compile the model
ensemble_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#history
history = ensemble_model.fit(train_dataset, epochs=50, validation_data=valid_dataset, shuffle=True)

#save model
ensemble_model.save('models/efficientNetB0_vgg16_FER.h5')

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('accuracy_plots/accuracy_efficientNetB0_vgg16_FER.png')