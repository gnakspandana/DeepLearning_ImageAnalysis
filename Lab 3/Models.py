import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt


def lenet_model(img_width, img_height, img_ch, base, n_class = 2):
    
    #block 1
    model1 = Sequential()
    model1.add(Conv2D(base, kernel_size = (3, 3), activation='relu',
                strides=1, padding='same',
                input_shape = (img_width, img_height, img_ch)))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    
    #block 2
    model1.add(Conv2D(base*2, kernel_size = (3, 3), activation='relu',
                strides=1, padding='same'))
    model1.add(MaxPooling2D(pool_size=(2, 2)))
    
    #dense layers start
    model1.add(Flatten())
    model1.add(Dense(base*2, activation='relu'))
    
    # if number of classes is 2, 1 neuron with sigmoid activation function will be the last dense layer.For more than 2 classes, n_class neuron with softmax will be used.
    if n_class == 2:
        model1.add(Dense(1))
        model1.add(Activation('sigmoid'))
    elif n_class > 2:
        model1.add(Dense(n_class))
        model1.add(Activation('softmax'))  
    
    model1.summary()
    
    return model1


def alexnet_model(img_width, img_height, img_ch, n_base, dropout = None, n_class = 2, first2_dense = [64, 64],
                  batch_norm = False, s_dropout = None):
   
    # block 1
    model1 = Sequential()
    model1.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch), kernel_size=(3,3),
    strides=(1,1), padding='same'))
    #if batch normalization is true, add batch normalization layer
    if batch_norm:
        model1.add(BatchNormalization())
    model1.add(Activation('relu'))
    #if user entered an s_dropout float number, spatial dropout will be active
    if isinstance(s_dropout, float):
        model1.add(SpatialDropout2D(s_dropout))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    
    # block 2
    model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    #if batch normalization is true, add batch normalization layer
    if batch_norm:
        model1.add(BatchNormalization())
    model1.add(Activation('relu'))
    #if user entered an s_dropout float number, spatial dropout will be active
    if isinstance(s_dropout, float):
        model1.add(SpatialDropout2D(s_dropout))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    
    #block 3
    model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    #if batch normalization is true, add batch normalization layer
    if batch_norm:
        model1.add(BatchNormalization())
    model1.add(Activation('relu'))    
    #if user entered an s_dropout float number, spatial dropout will be active    
    if isinstance(s_dropout, float):
        model1.add(SpatialDropout2D(s_dropout))
    
                   
    #block 4    
    model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    if batch_norm:
    #if batch normalization is true, add batch normalization layer
        model1.add(BatchNormalization())
    model1.add(Activation('relu'))
    #if user entered an s_dropout float number, spatial dropout will be active    
    if isinstance(s_dropout, float):
        model1.add(SpatialDropout2D(s_dropout))
    
                   
    #block 5
    model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    #if batch normalization is true, add batch normalization layer
    if batch_norm:
        model1.add(BatchNormalization())
    model1.add(Activation('relu'))
    #if user entered an s_dropout float number, spatial dropout will be active  
    if isinstance(s_dropout, float):
        model1.add(SpatialDropout2D(s_dropout))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    
    model1.add(Flatten())
    model1.add(Dense(first2_dense[0]))
    #if user entered a dropout float number,  dropout will be active  
    if isinstance(dropout, float):
        model1.add(Dropout(dropout))
    model1.add(Activation('relu'))
    
    model1.add(Dense(first2_dense[1]))
    #if user entered a dropout float number,  dropout will be active  
    if isinstance(dropout, float):
        model1.add(Dropout(dropout))
    model1.add(Activation('relu'))
    
    # if number of classes is 2, 1 neuron with sigmoid activation function will be the last dense layer.For more than 2 classes, n_class neuron with softmax will be used.
    if n_class == 2:
        model1.add(Dense(1))
#         if isinstance(dropout, float):
#             model1.add(Dropout(dropout))
    
        model1.add(Activation('sigmoid'))
    elif n_class > 2:
        model1.add(Dense(n_class))
#         if isinstance(dropout, float):
#             model1.add(Dropout(dropout))
    
        model1.add(Activation('softmax'))  
    
    model1.summary()
    
    return model1


def vgg16_model(img_width, img_height, img_ch, n_base, dropout = None, first2_dense = [64, 64], n_class = 2):
   
    model1 = Sequential()
    
    #Block1 - double convolutional layers with 3x3 kernel size and base number of feature maps
    model1.add(Conv2D(filters=n_base, input_shape=(img_width, img_height, img_ch), kernel_size=(3,3),
    strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))    
    model1.add(MaxPooling2D(pool_size=(2,2)))
   
    #Block2 - double convolutional layers with 3x3 kernel size and base*2 number of feature maps
    model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    
    #Block3 - triple convolutional layers with 3x3 kernel size and base*4 number of feature maps
    model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    
    #Block4 - triple convolutional layers with 3x3 kernel size and base*8 number of feature maps
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    
    #Block5 - triple convolutional layers with 3x3 kernel size and base*8 number of feature maps
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(Conv2D(filters= n_base *8, kernel_size=(3,3), strides=(1,1), padding='same'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    
    model1.add(Flatten())
    model1.add(Dense(first2_dense[0]))
    if isinstance(dropout, float):
        model1.add(Dropout(dropout))
    model1.add(Activation('relu'))
    
    model1.add(Dense(first2_dense[1]))
    if isinstance(dropout, float):
        model1.add(Dropout(dropout))
    model1.add(Activation('relu'))
    
    # if number of classes is 2, 1 neuron with sigmoid activation function will be the last dense layer.For more than 2 classes, n_class neuron with softmax will be used.
    if n_class == 2:
        model1.add(Dense(1))
#        if isinstance(dropout, float):
#            model1.add(Dropout(dropout))
    
        model1.add(Activation('sigmoid'))
    elif n_class > 2:
        model1.add(Dense(n_class))
#        if isinstance(dropout, float):
#            model1.add(Dropout(dropout))
    
        model1.add(Activation('softmax'))        
    
    model1.summary()
    
    return model1