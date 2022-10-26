import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, EfficientNetB0
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
                                    #zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale = 1./255,validation_split = 0.2)

test_datagen  = ImageDataGenerator(rescale = 1./255)

train_dataset  = train_datagen.flow_from_directory(directory = '../../input/fer2013/train',
                                                   target_size = (48,48),
                                                   class_mode = 'categorical',
                                                   subset = 'training',
                                                   batch_size = 64)
                                    
valid_dataset = valid_datagen.flow_from_directory(directory = '../../input/fer2013/train',
                                                  target_size = (48,48),
                                                  class_mode = 'categorical',
                                                  subset = 'validation',
                                                  batch_size = 64)
                                    
test_dataset = test_datagen.flow_from_directory(directory = '../../input/fer2013/test',
                                                  target_size = (48,48),
                                                  class_mode = 'categorical',
                                                  batch_size = 64)                            

IMAGE_SIZE = [48, 48]
eff_model = EfficientNetB0(input_shape=IMAGE_SIZE + [3],include_top=False,weights="imagenet")
vgg16_model = VGG16(input_shape=IMAGE_SIZE + [3],include_top=False,weights="imagenet")

for layer in eff_model.layers[:-4]:
    layer.trainable=False

for layer in vgg16_model.layers[:-4]:
    layer.trainable=False

x = Flatten()(eff_model.output)
x = Flatten()(vgg16_model.output)
prediction = Dense(7, activation='softmax')(x)
model = Model(inputs=eff_model.input, outputs=prediction)

model.summary()

plot_model(model, to_file='efficientNetB0_vgg16_FER.png', show_shapes=True,show_layer_names=True)
Image(filename='efficientNetB0_vgg16_FER.png') 

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_dataset,validation_data=valid_dataset,epochs = 50,verbose = 1)

#save model
model.save('efficientNetB0_vgg16_FER.h5')

plt.plot(history.history['accuracy'], label='accuracy')
plt.legend()
plt.show()
plt.savefig('accuracy_efficientNetB0_vgg16_FER.png')