# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""


import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from wettbewerb import load_references

# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

# ToDo: Update requirements.txt at the end of project
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Dropout, MaxPool1D, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import keras
from scipy import stats
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


image_size = 128
IMAGE_SIZE = [image_size, image_size]

# DONE: preprocess data into images and train the model with image data instead of time series data
# DONE: 1st attempt: simple images of ecg signal
# ToDo: 2nd attempt: convert ecg signal via STFT into images ToDo: test accuracy with different colorcodings)
# ToDo: 3rd attempt: convert ecg signal via Wavelet-transformation into images
def train_images():
    '''
    source: https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network./blob/master/model.py
    '''
    filepath = 'model_images'  #input("Enter the filename you want your model to be saved as: ")
    train_path = '../training_images_20' #input("Enter the directory of the training images: ")
    valid_path = '../test_images_20' #input("Enter the directory of the validation images: ")

    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    batch_size = 32

    model = create_model()
    print(model.summary())

    train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2) # rescale=1./255 to scale colors to values between [0,1]
    test_gen = ImageDataGenerator(rescale=1./255)

    train_generator = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size,subset='training')
    valid_generator = train_gen.flow_from_directory(train_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size,subset='validation')
    test_generator = test_gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE, shuffle=True, batch_size=batch_size)
    callbacks_list = [checkpoint]

    # ToDo: find right length of trining data,
    trainings_samples = train_generator.samples
    validation_samples = valid_generator.samples

    """
    WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least 
    `steps_per_epoch * epochs` batches (in this case, 557300 batches). You may need to use the repeat() function when building your dataset.
    
    WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least 
    `steps_per_epoch * epochs` batches (in this case, 1238 batches). You may need to use the repeat() function when building your dataset.
    
     W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    
    2021-11-16 12:27:12.944093: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    """
    r = model.fit(train_generator, validation_data=valid_generator, epochs=10,
                  steps_per_epoch=trainings_samples // batch_size, validation_steps=validation_samples // batch_size)     # epochs=50, callbacks=callbacks_list,

    return r, model

'''
source: https://github.com/daimenspace/ECG-arrhythmia-classification-using-a-2-D-convolutional-neural-network./blob/master/model.py
'''
# ToDo: search in literature for suitable model architectures
# Build the model
def create_model():
    # 2d cnn model for ecg image data
    model = Sequential()

    # We are using 4 convolution layers for feature extraction
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=[image_size, image_size, 3], kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    # ToDo: consider using Dropout layers to prevent overfitting
    #model.add(Dropout(0.2))  # This is the dropout layer. It's main function is to inactivate 20% of neurons in order to prevent overfitting

    model.add(Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))   # We use MaxPooling with a filter size of 2x2. This contributes to generalization
    model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform')) # , kernel_size=32, padding='same', kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # The prevous step gives an output of multi dimentional data, which cannot be fead directly into the feed forward neural network. Hence, the model is flattened
    model.add(Flatten())
    # One hidden layer of 2048 neurons have been used in order to have better classification results    # ToDo: compare classification results for different sizes of hidden layer
    model.add(Dense(2048))  # , kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.ELU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # The final neuron HAS to be of the same number as classes to predict and cannot be more than that.
    model.add(Dense(4, activation='softmax'))  # , activation='sigmoid')) # ToDo: update number of classes

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig
    start_time = time.time()
    history, model = train_images()
    pred_time = time.time() - start_time
    print("Runtime", pred_time, "s")

    #ToDo: save trained model in .npy format
    model.save('saved_model/my_model')

    # Source: https://github.com/kelvintanidi/Sign-Language-Detection/blob/master/Sign_Language_Detection.ipynb
    # save the models and weight for future purposes
    # serialize model to JSON
    model_json = model.to_json()
    with open("saved_model_json/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("saved_model_json/model.h5")
    print("Saved model to disk")

    # Plot the model Accuracy graph (Ideally, it should be Logarithmic shape)
    plt.plot(history.history['accuracy'],'r',linewidth=3.0, label='Training Accuracy')
    plt.plot(history.history['val_accuracy'],'b',linewidth=3.0, label='Testing Accuracy')
    plt.legend(fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

    # Plot the model Loss graph (Ideally it should be Exponentially decreasing shape)
    plt.plot(history.history['loss'], 'g', linewidth=3.0, label='Training Loss')
    plt.plot(history.history['val_loss'], 'y', linewidth=3.0, label='Testing Loss')
    plt.legend(fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
