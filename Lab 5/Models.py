# Import the required libraries
import os
import tensorflow as tf
import numpy as np
from random import shuffle
import cv2
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding2D, Activation, BatchNormalization, Input, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

import matplotlib.pyplot as plt


# Convolution Block Function
def conv_block(x, base = 8, kernel_size=(3, 3), activation='relu', batchnorm=False):
    
    #first convolutional layer
    x = Conv2D(base, kernel_size, padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    #second convolutional layer
    x = Conv2D(base, kernel_size, padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    return x

# U-Net Model Architecture
def get_unet(input_shape=(256, 256, 1), base = 8, kernel_size=(3, 3), activation='relu', batchnorm=False, dropout = None, n_class=2):
    
    inputs = Input(input_shape)

    # Encoder (contracting path)
    # Block-1
    conv1 = conv_block(inputs, base = base, kernel_size = kernel_size, activation = activation, batchnorm = batchnorm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if isinstance(dropout, float):  
        pool1 = Dropout(dropout)(pool1)
        
    # Block-2
    conv2 = conv_block(pool1, base = base*2, kernel_size = kernel_size, activation = activation, batchnorm = batchnorm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if isinstance(dropout, float):  
        pool2 = Dropout(dropout)(pool2)
    
    # Block-3
    conv3 = conv_block(pool2, base = base*4, kernel_size = kernel_size, activation = activation, batchnorm = batchnorm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if isinstance(dropout, float):  
        pool3 = Dropout(dropout)(pool3)
    
    # Block-4
    conv4 = conv_block(pool3, base = base*8, kernel_size = kernel_size, activation = activation, batchnorm = batchnorm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    if isinstance(dropout, float):  
        pool4 = Dropout(dropout)(pool4)

    # Block-5 Bottleneck
    conv5 = conv_block(pool4, base = base*16, kernel_size = kernel_size, activation = activation, batchnorm = batchnorm)

    # Decoder (expansion path)
    # Block-6 
    up1 = Conv2DTranspose(base*8, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat1 = concatenate([up1, conv4], axis=3)  
    if isinstance(dropout, float):  
        concat1 = Dropout(dropout)(concat1)
    conv6 = conv_block(concat1, base = base*8, kernel_size = kernel_size, activation = activation, batchnorm = batchnorm)
    
    # Block-7
    up2 = Conv2DTranspose(base*4, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat2 = concatenate([up2, conv3], axis=3)  
    if isinstance(dropout, float):  
        concat2 = Dropout(dropout)(concat2)
    conv7 = conv_block(concat2, base = base*4, kernel_size = kernel_size, activation = activation, batchnorm = batchnorm) 
    
    # Block-8
    up3 = Conv2DTranspose(base*2, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat3 = concatenate([up3, conv2], axis=3)  
    if isinstance(dropout, float):  
        concat3 = Dropout(dropout)(concat3)
    conv8 = conv_block(concat3, base = base*2, kernel_size = kernel_size, activation = activation, batchnorm = batchnorm)     
    
    # Block-9
    up4 = Conv2DTranspose(base, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat4 = concatenate([up4, conv1], axis=3)  
    if isinstance(dropout, float):  
        concat4 = Dropout(dropout)(concat4)
    conv9 = conv_block(concat4, base = base, kernel_size = kernel_size, activation = activation, batchnorm = batchnorm)         
        
    # Output layer
    if n_class == 2:
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    elif n_class > 2:
        conv10 = Conv2D(n_class, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def vgg16_model(img_width, img_height, img_ch, n_base = 64, dropout = None, n_class = 2, dense_layers = [64]):
   
    model1 = Sequential()
    
    #Block1 - double convolutional layers with 3x3 kernel size and base number of feature maps
    model1.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch), kernel_size=(3,3),
    strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))    
    model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
   
    #Block2 - double convolutional layers with 3x3 kernel size and base*2 number of feature maps
    model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    #Block3 - triple convolutional layers with 3x3 kernel size and base*4 number of feature maps
    model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    #Block4 - triple convolutional layers with 3x3 kernel size and base*8 number of feature maps
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    #Block5 - triple convolutional layers with 3x3 kernel size and base*8 number of feature maps
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same', name = 'Last_ConvLayer'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    
    model1.add(Flatten())
    #Dense Layers
    for dense_layer in dense_layers:
        model1.add(Dense(dense_layer))
        if isinstance(dropout, float):
            model1.add(Dropout(dropout))
        model1.add(Activation('relu'))
    
    #Last dense layer for classification
    model1.add(Dense(n_class))
    model1.add(Activation('softmax'))        
    
    model1.summary()
    
    return model1


def MLP_model(base = 128, dropout = None, n_class = 2):
    model1 = Sequential()
    
    #Dense Layer - 1
    model1.add(Flatten())
    model1.add(Dense(base)) 
    if isinstance(dropout, float):
        model1.add(Dropout(dropout))
    model1.add(Activation('relu'))
    
    #Dense Layer - Classification Layer
    model1.add(Dense(n_class))
    model1.add(Activation('softmax'))
    
    
    return model1