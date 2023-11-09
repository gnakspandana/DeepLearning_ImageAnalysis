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
import pickle


from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import psutil
from datetime import datetime
from timeit import default_timer as timer
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding2D, Activation, BatchNormalization, Input, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import SGD, Adam, RMSprop


from tensorflow.keras import applications

import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean as mean_calc

from Models import vgg16_model, MLP_model

    
def dict_of_models(best_models_path, folder_name):
    
    """
    DESCRIPTION: Function for loading all the models and storing them in a dictionary.
    ---INPUTS---
    best_models_path(string) = folder path that contains all of the folders with best models saved during hyperparameter tuning
    folder_name(string) = folder name of specific hyperparameter tuning operation which saved the best models in it
    
    ---OUTPUT---
    model_dict(dictionary) = a dictionary containing best models from each setting, fold combination
    """
    model_folder = os.path.join(best_models_path, folder_name)
    model_name_list = os.listdir(model_folder)
    
    if ".ipynb_checkpoints" in model_name_list:
        model_name_list.remove(".ipynb_checkpoints")
    model_name_list.sort()
    
    model_dict = {}
    
    for model_name in model_name_list:
        if model_name[-7] == "1":
            print("Models for setting " + model_name[8:10] + " is being loaded...")
        key = "setting_" + model_name[8:10] + "_fold_" + model_name[-7]
        model_dict[key] = tf.keras.models.load_model(os.path.join(model_folder,model_name))
        
    return model_dict

def dict_of_histories(model_histories_path, folder_name):
    
    """
    DESCRIPTION: Function for loading all the histories and storing them in a dictionary.
    ---INPUTS---
    model_histories_path(string) = folder path that contains all of the folders with training histories saved during hyperparameter tuning
    folder_name(string) = folder name of specific hyperparameter tuning operation which saved the best models in it
    
    ---OUTPUT---
    history_dict(dictionary) = a dictionary containing training histories from each setting, fold combination
    """
    
    history_folder = os.path.join(model_histories_path, folder_name)
    history_name_list = os.listdir(history_folder)
    
    if ".ipynb_checkpoints" in history_name_list:
        history_name_list.remove(".ipynb_checkpoints")
    history_name_list.sort()
    
    history_dict = {}
    
    for history_name in history_name_list:
        if history_name[-1] == "1":
            print("Histories for setting " + history_name[8:10] + " is being loaded...")
        key = "setting_" + history_name[8:10] + "_fold_" + history_name[-1]
        
        with open(os.path.join(history_folder,history_name), "rb") as history_file:
            history_dict[key] = pickle.load(history_file)
        
    return history_dict  



def plot_results(histories, title_list, figures_path, trends = 'both' ):
    
    """
    DESCRIPTION: Function for plotting trends of each metric based on the values provided in history files.
    ---INPUTS---
    histories(list)= = list of history files
    title_list(list) = list of titles for corresponding to history files
    figures_path(string) = the path where the generated figures will be saved to
    trends(string) = setting to decide on which trends should be drawn.  "both" is default value.
    -"train" draws only training data trends
    -"val" draws only validation data trends
    -"both" draws both training and validation data trends. 
    

    """
    
    if trends == 'train':
        metrics=["loss", "auc", "binary_accuracy", "recall", "precision"]
    elif trends == 'val':
        metrics=["val_loss", "val_auc",  "val_binary_accuracy", "val_recall", "val_precision"]     
    else: 
        if trends != 'both':
            print("trends not specified correctly, defaulting to using both")
            trends='both'
           
        metrics=["loss","auc",  "binary_accuracy", "recall", "precision"]
        metrics2=["val_loss", "val_auc",  "val_binary_accuracy", "val_recall", "val_precision"]       

    
    for j,clf_hist in enumerate(histories):
        #Prepare plots
        fig, axs = plt.subplots(1, len(metrics), figsize=(20, 4))
        fig.suptitle(title_list[j], fontsize=15)
        for i in range(len(metrics)):
            axs[i].set_title(metrics[i])
            axs[i].plot(clf_hist[metrics[i]], label=metrics[i])
            if trends == 'both':
                axs[i].plot(clf_hist[metrics2[i]], label=metrics2[i])
                if metrics2[i] == "val_loss":
                    axs[i].plot( np.argmin(clf_hist[metrics2[i]]),
                            np.min(clf_hist[metrics2[i]]),
                            marker="x", color="r", label="best model")
                else:
                    axs[i].plot( np.argmax(clf_hist[metrics2[i]]),
                            np.max(clf_hist[metrics2[i]]),
                            marker="x", color="r", label="best model")                    

            axs[i].set_xlabel("Epochs")
            #axs[i].set_ylabel("value")
            axs[i].legend()
        plt.savefig(os.path.join(figures_path,title_list[j] + '.jpeg'),  bbox_inches='tight')    
        #fig.title(title, fontsize = 25)
        #plt.show()
        
    
    plt.show()
    
    
    
def plot_roc(y_val,y_pred_list, legend_list, title, figures_path):
    
    """
    DESCRIPTION: Function for plotting Area under curves based on the predictions made by models.
    ---INPUTS---
    y_val(array) = labels of validation data
    y_pred_list(list) = list of predicted probabilities on validation data. Each value in list is contains predictions coming from a model.
    legend_list(list) = list of legends. Used to give model information in legend part of the plot.
    title(string) = title for the figure
    figures_path(string) = the path where the generated figures will be saved to
    """
    
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    


    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    
    for i,y_pred in enumerate(y_pred_list):
        fpr, tpr, thresholds_keras = roc_curve(y_val, y_pred)
        auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=legend_list[i] + ' (AUC = {:.3f})'.format(auc_value))
    
    plt.xlabel('False positive rate',  fontsize = 15)
    plt.ylabel('True positive rate', fontsize = 15)
    plt.title(title, fontsize = 20)
    plt.legend(loc='lower right', fontsize = 15)
    plt.savefig(os.path.join(figures_path,title+ '.jpeg'),  bbox_inches='tight')    
    plt.show()
    
def get_sec(time_str):
    """
    DESCRIPTION: Function for converting time data to seconds in integer
    ---INPUTS---
    time_str:Time data
    ---OUTPUTS---
    conversion result in seconds
    """
    
    #Get seconds from time.
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)   

# Load and preprocess images and masks
def load_data(image_dir, img_width, img_height, number_of_images=None, color = True):
    """
    DESCRIPTION: Function for loading image dataset fully or partially
    ---INPUTS---
    image_dir(string): Imageset directory
    img_width(int): width of the image
    img_height(int): height of the image
    number_of_images(int): Number of images to include in loading operation. None is default value which means loading all data.
    color(boolean): Boolean value for whether loading colored image or greyscale image
    ---OUTPUTS---
    images(list): a list of images found in the directory
    """
    if number_of_images == None:
        image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)] 
    elif isinstance(number_of_images, int):
        if len(os.listdir(image_dir)) > number_of_images:
            image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)[:number_of_images]] 
        else:
            image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)] 
            
    images = [load_image(img, img_width, img_height, color = color) for img in image_paths]


    return images



def load_image(image_path, img_width, img_height, color = True):
    """
    DESCRIPTION: Function for loading an image data 
    ---INPUTS---
    image_path(string): Path for the image data
    img_width(int): width of the image
    img_height(int): height of the image
    color(boolean): Boolean value for whether loading colored image or greyscale image
    ---OUTPUTS---
    image(matrix): a matrix of image data (normalized)
    """
    import cv2
    if color:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_width, img_height))
    #image normalization
    image = image / 255.0
    
    return image


def distribution_check(k,y):
    """
    DESCRIPTION: Function for checking the label distribution in the labels of each fold
    ---INPUTS---
    k(int):number of folds
    y(array): array of labels

    """
    fold_size = y.shape[0]//k
    
    for i in range(k):
        if i == k-1:
            data_portion = y[i*fold_size:]
        else:
            data_portion = y[i*fold_size:(i+1)*fold_size]
        print("fold {} has {} zeros and {} ones".format(i+1,(data_portion == 0).shape[0], (data_portion == 1).shape[0] ))
        
def train_val_folds(X,y,k):
    """
    DESCRIPTION: Function for preparing training folds and validation fold for each cross-validation iteration
    ---INPUTS---
    X(array) = array of image data
    y(array) = array of image labels
    k(int):number of folds

    ---OUTPUTS---
    X_train_folds(list): a list of training folds for each cross-validation iteration. Each value in the list contains the array of images 
    corresponding to training data of cross-validation iteration. 
    For example:
    -X_train_folds[0] = 2nd + 3rd folds are training data
    -X_train_folds[1] = 1st + 3rd folds are training data
    -X_train_folds[2] = 1st + 2nd folds are training data
    
    X_val_folds(list): a list of validation folds for each cross-validation iteration. Each value in the list contains the array of images
    corresponding to validation data of cross-validation iteration. 
    For example:
    -X_val_folds[0] = 1st fold is validation data
    -X_val_folds[1] = 2nd fold is validation data
    -X_val_folds[2] = 3rd fold is validation data
    
    y_train_folds(list): a list of labels corresponding to image arrays in X_train_folds.
    y_val_folds(list):  a list of labels corresponding to image arrays in X_val_folds.
    
    """
    fold_size = X.shape[0]//k
    indices = list(range(X.shape[0]))
    indices_of_folds = []
    
    X_train_folds = []
    X_val_folds = []
    y_train_folds = []
    y_val_folds = []
    
    #preparing only indices for each fold which will be used on indexing the provided data
    for i in range(k):
        if i == k-1:
            fold_indice = indices[i*fold_size:]
        else:
            fold_indice = indices[i*fold_size:(i+1)*fold_size]
        indices_of_folds.append(fold_indice)
    #preparing image and label arrays for each cross-validation iteraion
    for fold in range(k):

        indices_of_folds_copy = indices_of_folds.copy()
        val_fold = indices_of_folds_copy.pop(fold)
        train_fold = indices_of_folds_copy
        train_fold = sum(train_fold, [])


        X_train_folds.append(X[train_fold])
        X_val_folds.append(X[val_fold])
       
        
        y_train_folds.append(y[train_fold])
        y_val_folds.append(y[val_fold])
        
        
        print("Loop {}: Training data size: {}, Validation data size: {}".format((fold+1), len(train_fold),len(val_fold) ))
    

    return X_train_folds, X_val_folds, y_train_folds, y_val_folds


def hyperparameter_tuner(X_train_folds, y_train_folds,X_val_folds, y_val_folds, paths, k, img_width, img_height, ch, n_base_list, dropout_list, dense_layers_list, 
                        n_class, l2_reg_list, loss, lr_list, act_func_list, metrics, epochs, batch_size, verbose= 0, summary = False):
    
    """
    DESCRIPTION: Function for training models with different hyperparameters with k-fold crossvalidation and saving models,histories and a tuning report.
    ---INPUTS---
    X_train_folds(list): a list of training folds for each cross-validation iteration. Each value in the list contains the array of images 
    corresponding to training data of cross-validation iteration. 
    
    y_train_folds(list): a list of labels corresponding to image arrays in X_train_folds.
    
    X_val_folds(list): a list of validation folds for each cross-validation iteration. Each value in the list contains the array of images
    corresponding to validation data of cross-validation iteration.
    
    y_val_folds(list):  a list of labels corresponding to image arrays in X_val_folds.
    
    paths(list): a list that contains paths in following order
    1-best models
    2-model histories
    3-training results
    
    k(int):number of folds
    img_width(int): width of the image
    img_height(int): height of the image
    ch(int): number of channels in images
    n_base_list(list): a list of values for number of neurons to be set in convolutional layers(n_base)
    dropout_list(list): a list of dropout rates
    dense_layers_list(list): a list of lists containing classification dense layer neurons. For example:
        -[128,128] : classificication layers will consists of 2 layers with 128 neurons each.
    n_class(int): number of classes in the dataset
    l2_reg_list(list): list of weight decays to be used in L2 regularization.
    loss(string): type of loss
    lr_list(list): list of learning rate
    act_func_list(list): a list of activation functions
    metrics(list): a list of metrics to be evaluated during training
    epochs(int): number of epochs chosen for training
    batch_size(int): size of the batch chosen for training
    verbose(int): verbose value for printing details. Default value is 0, which means nothing will be printed
    summary(boolean): boolean for printing summary of created models.Default value is False.
    
    ---OUTPUT---
    report_df(pandas dataframe) = a report containg setting values, training times, and metric values calculated during each cross-validation iteration.
    """
    
    best_models_path = paths[0]
    model_histories_path = paths[1]
    training_results_path = paths[2]
    
    now = datetime.now()
    current_time = now.strftime("%d-%m-%Y_%H-%M-%S")
    print("Start time for hyperparameter tuning operation =", current_time)
    
    best_models_folder = os.path.join(best_models_path, "best_models_session_" + current_time)
    model_histories_folder = os.path.join(model_histories_path, "model_histories_session_" + current_time)
    
    os.mkdir(best_models_folder)
    print("best models directory for this training session is created " + best_models_folder)
    os.mkdir(model_histories_folder)
    print("model histories directory for this training session is created " + model_histories_folder)
    
    report_file_name = "Hyperparameter_tuning_report_"  + current_time
    setting_count = 0
    #columns of the report is prepared
    report_df = pd.DataFrame(columns = ["Setting_number","img_width", "img_height", "ch", \
                                        "n_base","dropout", "dense_layers", "n_class", "l2_reg", "act_func",\
                                        "loss_type", "learning_rate", "epochs", "batch_size",\
                                        "CV1_best_loss_epoch","CV2_best_loss_epoch","CV3_best_loss_epoch",\
                                        "CV1_best_auroc_epoch","CV2_best_auroc_epoch","CV3_best_auroc_epoch",\
                                        "CV1_train_time", "CV2_train_time","CV3_train_time",\
                                        "CV1_val_loss", "CV2_val_loss", "CV3_val_loss", \
                                        "CV1_val_acc", "CV2_val_acc", "CV3_val_acc", \
                                        "CV1_val_auroc", "CV2_val_auroc", "CV3_val_auroc", \
                                        "CV1_val_precision", "CV2_val_precision", "CV3_val_precision", \
                                        "CV1_val_recall", "CV2_val_recall", "CV3_val_recall", \
                                        "CV_avg_val_acc", "CV_avg_val_auroc", "CV_avg_val_precision","CV_avg_val_recall"])
    main_train_start = timer()
    for n_base in n_base_list:
        for dropout in dropout_list:
            for dense_layers in dense_layers_list:
                for l2_reg in l2_reg_list:
                    for learning_rate in lr_list:
                        for act_func in act_func_list:
                            setting_count += 1 #used for counting which setting is being trained
                            #lists to store values found in each cross-validation iteration
                            cv_best_loss_epoch_list = []
                            cv_best_auroc_epoch_list = []
                            cv_train_time_list = []
                            cv_val_loss_list = []
                            cv_val_acc_list = []
                            cv_val_auroc_list = []
                            cv_val_precision_list = []
                            cv_val_recall_list = []
                            for fold in range(k):
                                """START OF A FOLD TRAINING"""


                                #callback for saving the best model based on validation auroc score.
                                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(best_models_folder,"setting_"+ str(setting_count) +"_best_model_fold_"+ str(fold+1) +".keras"),
                                                                                 monitor='val_auc',
                                                                                 save_best_only = True,
                                                                                 mode = "max",
                                                                                 save_weights_only=False,
                                                                                 verbose=verbose)


                                clf = vgg16_model(img_width, img_height, ch, n_base, dropout = dropout, dense_layers = dense_layers, act_func = act_func, n_class = n_class, l2_reg = l2_reg, summary= summary)

                                clf.compile(loss=loss, optimizer = Adam(learning_rate = learning_rate), metrics=metrics) 
                                
                                #timer for saving how long the training takes
                                cv_train_start = timer()

                                clf_hist = clf.fit(X_train_folds[fold], y_train_folds[fold], shuffle=False, epochs = epochs, batch_size=batch_size, verbose=verbose, 
                                                       validation_data=(X_val_folds[fold], y_val_folds[fold]),  callbacks=[cp_callback])

                                cv_train_end = timer()
                                #saving training history
                                with open(os.path.join(model_histories_folder,"setting_"+ str(setting_count) +"_training_history_fold_"+ str(fold+1)), 'wb') as history_file:
                                    pickle.dump(clf_hist.history, history_file)

                                cv_best_loss_epoch_list.append(np.argmin(clf_hist.history["val_loss"]) + 1)
                                cv_best_auroc_epoch_list.append(np.argmax(clf_hist.history["val_auc"]) + 1)
                                cv_train_time_list.append(time.strftime('%H:%M:%S', time.gmtime(cv_train_end - cv_train_start)))
                                cv_val_loss_list.append(np.min(clf_hist.history["val_loss"]))
                                cv_val_acc_list.append(np.max(clf_hist.history["val_binary_accuracy"]))
                                cv_val_auroc_list.append(np.max(clf_hist.history["val_auc"]))
                                cv_val_precision_list.append(np.max(clf_hist.history["val_precision"]))
                                cv_val_recall_list.append(np.max(clf_hist.history["val_recall"]))

                            #creating a list of all metrics calculated in this training section
                            tuning_result = [setting_count, img_width, img_height,ch, n_base, \
                                             dropout, dense_layers, n_class, l2_reg, act_func, loss,\
                                             learning_rate, epochs, batch_size, \
                                             cv_best_loss_epoch_list[0],cv_best_loss_epoch_list[1], cv_best_loss_epoch_list[2],\
                                             cv_best_auroc_epoch_list[0],cv_best_auroc_epoch_list[1], cv_best_auroc_epoch_list[2],\
                                             cv_train_time_list[0],cv_train_time_list[1], cv_train_time_list[2],\
                                             cv_val_loss_list[0],cv_val_loss_list[1], cv_val_loss_list[2],\
                                             cv_val_acc_list[0],cv_val_acc_list[1], cv_val_acc_list[2],\
                                             cv_val_auroc_list[0],cv_val_auroc_list[1], cv_val_auroc_list[2],\
                                             cv_val_precision_list[0],cv_val_precision_list[1], cv_val_precision_list[2],\
                                             cv_val_recall_list[0],cv_val_recall_list[1], cv_val_recall_list[2],\
                                             mean_calc(cv_val_acc_list), mean_calc(cv_val_auroc_list), \
                                             mean_calc(cv_val_precision_list), mean_calc(cv_val_recall_list)]
                            #adding the training results as a row
                            report_df.loc[len(report_df)] = tuning_result
                            #writing the results to report file after each setting training.
                            report_df.to_csv(os.path.join(training_results_path, report_file_name + ".txt"), sep = "\t", index = False)   
                            if setting_count % 5 == 1: 
                                print(str(setting_count) + " settings have been checked and saved to report file")
                            #used to check the system memory 
                            print(psutil.virtual_memory())
    
    
    report_df.to_csv(os.path.join(training_results_path, report_file_name + ".txt"), sep = "\t", index = False)

    main_train_end_date = datetime.now()
    print("Training end date is:", main_train_end_date)

    main_train_end = timer() 
    main_train_time = main_train_end - main_train_start

    print("Total Training Time is:", time.strftime("%d-%m-%Y_%H-%M-%S", time.gmtime(main_train_time)))
    return report_df


def pretrained_hyperparameter_tuner(X_train_folds, y_train_folds,X_val_folds, y_val_folds, img_width, img_height, paths, k, dropout_list, dense_layers_list, 
                        n_class, l2_reg_list, loss, lr_list, act_func_list, metrics, epochs, batch_size, verbose= 0, summary = False):
    
    """
    DESCRIPTION: Function for training MLP models using pretrained VGG16 with different hyperparameters with k-fold crossvalidation and saving models,histories and a tuning report.
    ---INPUTS---
    X_train_folds(list): a list of training folds for each cross-validation iteration. Each value in the list contains the array of images 
    corresponding to training data of cross-validation iteration. 
    
    y_train_folds(list): a list of labels corresponding to image arrays in X_train_folds.
    
    X_val_folds(list): a list of validation folds for each cross-validation iteration. Each value in the list contains the array of images
    corresponding to validation data of cross-validation iteration.
    
    y_val_folds(list):  a list of labels corresponding to image arrays in X_val_folds.
    
    img_width(int): width of the image
    img_height(int): height of the image
    paths(list): a list that contains paths in following order
    1-best models
    2-model histories
    3-training results
    
    k(int):number of folds

    dropout_list(list): a list of dropout rates
    dense_layers_list(list): a list of lists containing classification dense layer neurons. For example:
        -[128,128] : classificication layers will consists of 2 layers with 128 neurons each.
    n_class(int): number of classes in the dataset
    l2_reg_list(list): list of weight decays to be used in L2 regularization.
    loss(string): type of loss
    lr_list(list): list of learning rate
    act_func_list(list): a list of activation functions
    metrics(list): a list of metrics to be evaluated during training
    epochs(int): number of epochs chosen for training
    batch_size(int): size of the batch chosen for training
    verbose(int): verbose value for printing details. Default value is 0, which means nothing will be printed
    summary(boolean): boolean for printing summary of created models. Default value is False.
    
    ---OUTPUT---
    report_df(pandas dataframe) = a report containg setting values, training times, and metric values calculated during each cross-validation iteration.
    """
    
    best_models_path = paths[0]
    model_histories_path = paths[1]
    training_results_path = paths[2]
    
    now = datetime.now()
    current_time = now.strftime("%d-%m-%Y_%H-%M-%S")
    print("Start time for hyperparameter tuning operation =", current_time)
    
    best_models_folder = os.path.join(best_models_path, "best_models_session_" + current_time)
    model_histories_folder = os.path.join(model_histories_path, "model_histories_session_" + current_time)
    
    os.mkdir(best_models_folder)
    print("best models directory for this training session is created " + best_models_folder)
    os.mkdir(model_histories_folder)
    print("model histories directory for this training session is created " + model_histories_folder)
    
    report_file_name = "Hyperparameter_tuning_report_"  + current_time
    setting_count = 0
    #columns of the report is prepared
    report_df = pd.DataFrame(columns = ["Setting_number","img_width", "img_height",  \
                                        "dropout", "dense_layers", "n_class", "l2_reg", "act_func",\
                                        "loss_type", "learning_rate", "epochs", "batch_size",\
                                        "CV1_best_loss_epoch","CV2_best_loss_epoch","CV3_best_loss_epoch",\
                                        "CV1_best_auroc_epoch","CV2_best_auroc_epoch","CV3_best_auroc_epoch",\
                                        "CV1_train_time", "CV2_train_time","CV3_train_time",\
                                        "CV1_val_loss", "CV2_val_loss", "CV3_val_loss", \
                                        "CV1_val_acc", "CV2_val_acc", "CV3_val_acc", \
                                        "CV1_val_auroc", "CV2_val_auroc", "CV3_val_auroc", \
                                        "CV1_val_precision", "CV2_val_precision", "CV3_val_precision", \
                                        "CV1_val_recall", "CV2_val_recall", "CV3_val_recall", \
                                        "CV_avg_val_acc", "CV_avg_val_auroc", "CV_avg_val_precision","CV_avg_val_recall"])
    main_train_start = timer()

    for dropout in dropout_list:
        for dense_layers in dense_layers_list:
            for l2_reg in l2_reg_list:
                for learning_rate in lr_list:
                    for act_func in act_func_list:
                        setting_count += 1 #used for counting which setting is being trained
                        #lists to store values found in each cross-validation iteration
                        cv_best_loss_epoch_list = []
                        cv_best_auroc_epoch_list = []
                        cv_train_time_list = []
                        cv_val_loss_list = []
                        cv_val_acc_list = []
                        cv_val_auroc_list = []
                        cv_val_precision_list = []
                        cv_val_recall_list = []
                        for fold in range(k):
                            """START OF A FOLD TRAINING"""


                            #callback for saving the best model based on validation auroc score.
                            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(best_models_folder,"setting_"+ str(setting_count) +"_best_model_fold_"+ str(fold+1) +".keras"),
                                                                             monitor='val_auc',
                                                                             save_best_only = True,
                                                                             mode = "max",
                                                                             save_weights_only=False,
                                                                             verbose=verbose)
                            #loading pretrained VGG16 model and outputing features it extracts from training and validation data
                            pretrained_vgg16_model = applications.VGG16(include_top=False, weights='imagenet')
                            X_train = pretrained_vgg16_model.predict( X_train_folds[fold])
                            X_val = pretrained_vgg16_model.predict( X_val_folds[fold])
                            
                            #MLP model for training on extracted features
                            clf = MLP_model(dense_layers = dense_layers, act_func = act_func,dropout = dropout, n_class = n_class, l2_reg = l2_reg, summary = summary)

                            clf.compile(loss=loss, optimizer = Adam(learning_rate = learning_rate), metrics=metrics) 

                            #timer for saving how long the training takes
                            cv_train_start = timer()

                            clf_hist = clf.fit(X_train, y_train_folds[fold], shuffle=False, epochs = epochs, batch_size=batch_size, verbose=verbose, 
                                                   validation_data=(X_val, y_val_folds[fold]),  callbacks=[cp_callback])

                            cv_train_end = timer()
                            #saving training history
                            with open(os.path.join(model_histories_folder,"setting_"+ str(setting_count) +"_training_history_fold_"+ str(fold+1)), 'wb') as history_file:
                                pickle.dump(clf_hist.history, history_file)

                            cv_best_loss_epoch_list.append(np.argmin(clf_hist.history["val_loss"]) + 1)
                            cv_best_auroc_epoch_list.append(np.argmax(clf_hist.history["val_auc"]) + 1)
                            cv_train_time_list.append(time.strftime('%H:%M:%S', time.gmtime(cv_train_end - cv_train_start)))
                            cv_val_loss_list.append(np.min(clf_hist.history["val_loss"]))
                            cv_val_acc_list.append(np.max(clf_hist.history["val_binary_accuracy"]))
                            cv_val_auroc_list.append(np.max(clf_hist.history["val_auc"]))
                            cv_val_precision_list.append(np.max(clf_hist.history["val_precision"]))
                            cv_val_recall_list.append(np.max(clf_hist.history["val_recall"]))

                        #creating a list of all metrics calculated in this training section
                        tuning_result = [setting_count, img_width, img_height, \
                                         dropout, dense_layers, n_class, l2_reg, act_func, loss,\
                                         learning_rate, epochs, batch_size, \
                                         cv_best_loss_epoch_list[0],cv_best_loss_epoch_list[1], cv_best_loss_epoch_list[2],\
                                         cv_best_auroc_epoch_list[0],cv_best_auroc_epoch_list[1], cv_best_auroc_epoch_list[2],\
                                         cv_train_time_list[0],cv_train_time_list[1], cv_train_time_list[2],\
                                         cv_val_loss_list[0],cv_val_loss_list[1], cv_val_loss_list[2],\
                                         cv_val_acc_list[0],cv_val_acc_list[1], cv_val_acc_list[2],\
                                         cv_val_auroc_list[0],cv_val_auroc_list[1], cv_val_auroc_list[2],\
                                         cv_val_precision_list[0],cv_val_precision_list[1], cv_val_precision_list[2],\
                                         cv_val_recall_list[0],cv_val_recall_list[1], cv_val_recall_list[2],\
                                         mean_calc(cv_val_acc_list), mean_calc(cv_val_auroc_list), \
                                         mean_calc(cv_val_precision_list), mean_calc(cv_val_recall_list)]
                        #adding the training results as a row
                        report_df.loc[len(report_df)] = tuning_result
                        #writing the results to report file after each setting training.
                        report_df.to_csv(os.path.join(training_results_path, report_file_name + ".txt"), sep = "\t", index = False)   
                        if setting_count % 5 == 1: 
                            print(str(setting_count) + " settings have been checked and saved to report file")
                        #used to check the system memory 
                        print(psutil.virtual_memory())

    
    report_df.to_csv(os.path.join(training_results_path, report_file_name + ".txt"), sep = "\t", index = False)

    main_train_end_date = datetime.now()
    print("Training end date is:", main_train_end_date)

    main_train_end = timer() 
    main_train_time = main_train_end - main_train_start

    print("Total Training Time is:", time.strftime("%H-%M-%S", time.gmtime(main_train_time)))
    return report_df        

