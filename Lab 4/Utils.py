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

def loss_accuracy_plot(clf_hist, string , accuracy = "binary_accuracy"):

#Function for plotting the results of loss and accuracy

#------INPUTS------
#clf_hist = classifier history during training.
#string = title for the plot.
#n_class = number of class in the classification task. Based on the value, the accuracy type changes.


    if accuracy == "binary_accuracy":
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        #Plot for loss values
        axs[0].set_title("Learning curve")
        axs[0].plot(clf_hist.history["loss"], label="loss")
        axs[0].plot(clf_hist.history["val_loss"], label="val_loss")
        axs[0].plot( np.argmin(clf_hist.history["val_loss"]),
                    np.min(clf_hist.history["val_loss"]),
                    marker="x", color="r", label="best model")

        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss Value")
        axs[0].legend()

        fig.suptitle(string, fontsize=13)

        #Plot for accuracy values
        axs[1].set_title("Learning curve")
        axs[1].plot(clf_hist.history["binary_accuracy"], label="accuracy")
        axs[1].plot(clf_hist.history["val_binary_accuracy"], label="val_accuracy")
        axs[1].plot( np.argmax(clf_hist.history["val_binary_accuracy"]),
                    np.max(clf_hist.history["val_binary_accuracy"]),
                    marker="x", color="r", label="best model")

        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        plt.show()
    
    elif accuracy == "sparse_categorical_accuracy":
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        #Plot for loss values
        axs[0].set_title("Learning curve")
        axs[0].plot(clf_hist.history["loss"], label="loss")
        axs[0].plot(clf_hist.history["val_loss"], label="val_loss")
        axs[0].plot( np.argmin(clf_hist.history["val_loss"]),
                    np.min(clf_hist.history["val_loss"]),
                    marker="x", color="r", label="best model")

        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss Value")
        axs[0].legend()

        fig.suptitle(string, fontsize=13)

        #Plot for accuracy values
        axs[1].set_title("Learning curve")
        axs[1].plot(clf_hist.history["sparse_categorical_accuracy"], label="accuracy")
        axs[1].plot(clf_hist.history["val_sparse_categorical_accuracy"], label="val_accuracy")
        axs[1].plot( np.argmax(clf_hist.history["val_sparse_categorical_accuracy"]),
                    np.max(clf_hist.history["val_sparse_categorical_accuracy"]),
                    marker="x", color="r", label="best model")

        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        plt.show()
    
  
def get_length(Path, Pattern):
    # Pattern: name of the subdirectory
    Length = len(os.listdir(os.path.join(Path, Pattern)))
    
    return Length


def pretrain_data_label(train_data_dir, validation_data_dir, img_width, img_height, batch_size, class_labels = ['AFF','NFF']):
    #This function extracts features output from pretrained VGG16 model 
    
    #INPUTS
    #train_data_dir = directory of training data
    #validation_data_dir = directory of validation data
    #img_width = image width
    #img_height = image height
    #batch_size = size of the batch
    #class_labels = a list of class labels of the chosen dataset. Default labels are for bone dataset.
    
    #OUTPUTS
    #train_data = training data (extracted features)
    #train_label = training labels
    #validation_data = validation data (extracted features)
    #validation_labels = validation labels
    
    
    # number of data for each class 
    Len_C1_Train = get_length(train_data_dir, class_labels[0])
    Len_C2_Train = get_length(train_data_dir, class_labels[1])
    Len_C1_Val = get_length(validation_data_dir, class_labels[0])
    Len_C2_Val = get_length(validation_data_dir, class_labels[1])
    
    
    # loading the pre-trained model
    # include top: false means that the dense layers at the top of the network will not be used. 
    model = applications.VGG16(include_top=False, weights='imagenet')
    model.summary()
    
    # Feature extraction from pretrained VGG (training data)
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(train_data_dir,
                                                 target_size=(img_width, img_height),
                                                 batch_size=batch_size,
                                                 class_mode="binary",
                                                 shuffle=False)
    # Extracting the features from the loaded images
    features_train = model.predict( train_generator,
                                              (Len_C1_Train+Len_C2_Train) // batch_size,  #Steps
                                             max_queue_size=1)
    
    
    # To DO: Feature extraction from pretrained VGG (validation data)
    # First create another generator 
    datagen_val = ImageDataGenerator(rescale=1. / 255)

    validation_generator = datagen_val.flow_from_directory(validation_data_dir,
                                                            target_size=(img_width, img_height),
                                                            batch_size=batch_size,
                                                            class_mode='binary',
                                                            shuffle=False)

    # Extracting features from the loaded validation images
    features_validation = model.predict(validation_generator,
                                        (Len_C1_Val + Len_C2_Val) // batch_size, #Steps
                                         max_queue_size=1)

    # training a small MLP with extracted features from the pre-trained model
    # In fact this MLP will be used instead of the dense layers of the VGG model 
    # and only this MLP will be trained on the dataset.

    train_data = features_train
    train_labels = np.array([0] * int(Len_C1_Train) + [1] * int(Len_C2_Train))

    validation_data = features_validation
    validation_labels = np.array([0] * int(Len_C1_Val) + [1] * int(Len_C2_Val))
    
    return train_data, train_labels, validation_data, validation_labels

def loss_dice_plot(clf_hist, string):

#Function for plotting the results of loss and dice coefficient

#------INPUTS------
#clf_hist = classifier history during training.
#string = title for the plot.

                
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    #Plot for loss values
    axs[0].set_title("Learning curve")
    axs[0].plot(clf_hist.history["loss"], label="loss")
    axs[0].plot(clf_hist.history["val_loss"], label="val_loss")
    axs[0].plot( np.argmin(clf_hist.history["val_loss"]),
                    np.min(clf_hist.history["val_loss"]),
                    marker="x", color="r", label="best model")

    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss Value")
    axs[0].legend()

    fig.suptitle(string, fontsize=13)

    #Plot for dice values
    axs[1].set_title("Learning curve")
    axs[1].plot(clf_hist.history["dice_coef"], label="dice_coefficient")
    axs[1].plot(clf_hist.history["val_dice_coef"], label="val_dice_coefficient")
    axs[1].plot( np.argmax(clf_hist.history["val_dice_coef"]),
                    np.max(clf_hist.history["val_dice_coef"]),
                    marker="x", color="r", label="best model")

    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Dice Coefficient")
    axs[1].legend()
    plt.show()
    
    
# Load and preprocess images and masks
def load_data(data_dir, image_dir, mask_dir, img_width, img_height):
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    mask_paths = [os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir)]

    images = [load_image(img, img_width, img_height) for img in image_paths]
    masks = [load_mask(mask, img_width, img_height) for mask in mask_paths]

    return images, masks

def load_image(image_path, img_width, img_height):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_width, img_height))
    image = image / 255.0
    return image

def load_mask(mask_path, img_width, img_height):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (img_width, img_height))
    mask = mask / 255.0
    return mask

#Functions for data augmentation
def create_generator(image, mask, generator, batch_size, seed=None):
    image_generator = generator.flow(image, batch_size=batch_size, seed=seed)
    mask_generator = generator.flow(mask, batch_size=batch_size, seed=seed)

    return image_generator, mask_generator

def image_mask_generator(image_generator, mask_generator):
    image_mask_generator_zip = zip(image_generator, mask_generator)
    for (img, mask) in image_mask_generator_zip:
        yield (img, mask)

#Dice coefficient
def dice_coef(y_true, y_pred, smooth = 1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#Dice loss
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Training Function
def train_unet(model, train_images, train_masks, val_images, val_masks, batch_size, epochs, learning_rate,
               metrics, verbose = 1,
               loss = 'binary_crossentropy', generator = None):
    
    if generator == None:
        model.compile(optimizer=Adam(learning_rate = learning_rate), loss=loss, metrics=metrics)

        # Train the model
        history = model.fit(train_images, train_masks, validation_data=(val_images, val_masks), verbose = verbose,
                            batch_size=batch_size, epochs=epochs)
    elif generator != None:
        train_image_generator, train_mask_generator = create_generator(train_images, train_masks, generator, batch_size,
                                                                       seed=None)
    
        train_image_mask_generator = image_mask_generator(train_image_generator, train_mask_generator)

        val_image_generator, val_mask_generator = create_generator(val_images, val_masks, generator, batch_size, seed=None)

        val_image_mask_generator = image_mask_generator(val_image_generator, val_mask_generator)



        model.compile(optimizer=Adam(learning_rate = learning_rate), loss=loss, metrics=metrics)

        step_size_train = train_image_generator.n//train_image_generator.batch_size
        step_size_val = val_image_generator.n//val_image_generator.batch_size

        history = model.fit_generator(generator = train_image_mask_generator,steps_per_epoch = step_size_train,
                                         epochs = epochs, verbose=verbose, 
                                           validation_data=val_image_mask_generator, validation_steps=step_size_val)
        
    return model, history