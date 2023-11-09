"""Lab Group 8:

- Gnana Spandana Akumalla
- Patrik Svensson
- Serkan Arda Yilal
"""
# Import the required libraries
import os
import tensorflow as tf
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding2D, Activation, BatchNormalization, Input, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow import keras

import matplotlib.pyplot as plt





def vgg16_model(img_width, img_height, img_ch, n_base = 64, dropout = None, n_class = 2, dense_layers = [64], act_func = "relu", l2_reg = None, summary= False):
    
    """
    DESCRIPTION: Function for creating a vgg16 model.
    ---INPUTS---

    img_width(int): width of the image
    img_height(int): height of the image
    img_ch(int): number of channels in an image
    n_base(int): number of neurons to be used in convolutional layers. Default value is 64
    dropout(float): a float value for dropout rate. Default value is None, which is no dropout.
    n_class(int): number of classes in the dataset. Default value is 2.
    dense_layers(list): a list of classification dense layer neurons. For example:
        -[128,128] : classificication layers will consists of 2 layers with 128 neurons each. Default value is [64]
    act_func(string): activation function to be used (except last layer)
    l2_reg(float): a float value for weight decay value to be used in L2 Regularization. Default value is None, which is no regularization.
    summary(boolean): boolean for printing summary of created models. Default value is False.

    
    
    ---OUTPUT---
    model1(vgg16 model) = a vgg16 model created by provided settings.
    """
    model1 = Sequential()
    
    if isinstance(l2_reg, float): # with l2 regularization
        #Block1 - double convolutional layers with 3x3 kernel size and base number of feature maps
        model1.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch), kernel_size=(3,3),
        strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))    
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
       
        #Block2 - double convolutional layers with 3x3 kernel size and base*2 number of feature maps
        model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        #Block3 - triple convolutional layers with 3x3 kernel size and base*4 number of feature maps
        model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        #Block4 - triple convolutional layers with 3x3 kernel size and base*8 number of feature maps
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        #Block5 - triple convolutional layers with 3x3 kernel size and base*8 number of feature maps
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same', name = 'Last_ConvLayer', kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
        model1.add(Activation(act_func))
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        
        model1.add(Flatten())
        #Dense Layers
        for dense_layer in dense_layers:
            model1.add(Dense(dense_layer, kernel_regularizer=keras.regularizers.l2(l=l2_reg) ))
            if isinstance(dropout, float):
                model1.add(Dropout(dropout))
            model1.add(Activation(act_func))
        
        #Last dense layer for classification
        if n_class == 2:
            model1.add(Dense(1))
            model1.add(Activation('sigmoid'))
        else:
            model1.add(Dense(n_class))
            model1.add(Activation('softmax'))
    else: # no regularization
        #Block1 - double convolutional layers with 3x3 kernel size and base number of feature maps
        model1.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch), kernel_size=(3,3),
        strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))    
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
       
        #Block2 - double convolutional layers with 3x3 kernel size and base*2 number of feature maps
        model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        #Block3 - triple convolutional layers with 3x3 kernel size and base*4 number of feature maps
        model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        #Block4 - triple convolutional layers with 3x3 kernel size and base*8 number of feature maps
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        #Block5 - triple convolutional layers with 3x3 kernel size and base*8 number of feature maps
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
        model1.add(Activation(act_func))
        model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same', name = 'Last_ConvLayer'))
        model1.add(Activation(act_func))
        model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        
        model1.add(Flatten())
        #Dense Layers
        for dense_layer in dense_layers:
            model1.add(Dense(dense_layer))
            if isinstance(dropout, float):
                model1.add(Dropout(dropout))
            model1.add(Activation(act_func))
        
        #Last dense layer for classification
        if n_class == 2:
            model1.add(Dense(1))
            model1.add(Activation('sigmoid'))
        else:
            model1.add(Dense(n_class))
            model1.add(Activation('softmax'))  
   
    if summary:
        model1.summary()
    
    return model1


def MLP_model(dense_layers = [128,128], act_func = "relu",dropout = None, n_class = 2, l2_reg = 0.0001, summary = False):
    """
    DESCRIPTION: Function for creating an MLP model to be used on features extracted by pretrained VGG16.
    ---INPUTS---
    dense_layers(list): a list of classification dense layer neurons. For example:
        -[128,128] : classificication layers will consists of 2 layers with 128 neurons each. Default value is [128,128]
    act_func(string): activation function to be used (except last layer)
    dropout(float): a float value for dropout rate. Default value is None, which is no dropout.
    n_class(int): number of classes in the dataset. Default value is 2.
    l2_reg(float): a float value for weight decay value to be used in L2 Regularization. Default value is None, which is no regularization.
    summary(boolean): boolean for printing summary of created models. Default value is False.

    
    
    ---OUTPUT---
    model1(MLP model) = an MLP model created by provided settings.
    """
    model1 = Sequential()
    
    
    model1.add(Flatten())
    
    for dense_layer in dense_layers:
        model1.add(Dense(dense_layer, kernel_regularizer=keras.regularizers.l2(l=l2_reg) )) 
        if isinstance(dropout, float):
            model1.add(Dropout(dropout))
        model1.add(Activation(act_func))
    

    if n_class == 2:
        model1.add(Dense(1))
        model1.add(Activation('sigmoid'))
    else:
        model1.add(Dense(n_class))
        model1.add(Activation('softmax'))
    
    if summary:
        model1.summary()
    
    
    return model1