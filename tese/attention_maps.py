import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix
import os

#visualize attention maps for a given image
def get_attention_map(model, layer_name, x):
    #get the output of the last convolutional layer
    last_conv_layer = model.get_layer(layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)
    
    #get the gradients of the top predicted class for our image with respect to the output feature map of the last conv layer
    with tf.GradientTape() as gtape:
        conv_outputs = last_conv_layer_model(x)
        predictions = model(x)
        top_pred_index = tf.argmax(predictions[0])
        top_conv_output = conv_outputs[0, :, :, top_pred_index]
    
    grads = gtape.gradient(top_conv_output, conv_outputs)[0]
    
    #pool the gradients over all the axises leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    #iterate over each of the channel and multiply each channel in the feature map array by how important this channel is with regard to the top predicted class
    #then sum all the channels to obtain the heatmap class activation
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    #for visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def visualize_attention_map(img_path, model, layer_name):
    #load the image
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    #get the attention map
    attention_map = get_attention_map(model, layer_name, x)
    
    #plot the attention map
    plt.imshow(attention_map)
    plt.show()
    
    #plot the image
    plt.imshow(img)
    plt.show()
    
    #plot the attention map over the image
    plt.imshow(img)
    plt.imshow(attention_map, alpha=0.5)
    plt.show()

#load the model
model = load_model('inceptionv3.h5')

#visualize attention maps for a given image
visualize_attention_map('test.jpg', model, 'mixed10')
