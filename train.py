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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import time

# ToDo: Update requirements.txt at the end of project
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Dropout, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers
from torch.nn.utils.rnn import pad_sequence
import torch


#def show_shapes(): # can make yours to take inputs; this'll use local variable values
#    print("Expected: (num_samples, timesteps, channels)")
#    print("Sequences: {}".format(ecg_leads_array.shape))
#    print("Targets:   {}".format(ecg_labels_array.shape))

# ToDo: preprocess data into images and train the model with image data instead of time series data
# ToDo: 1st attempt: simple images of ecg signal
# ToDo: 2nd attempt: convert ecg signal via FFT into images
# ToDo: 3rd attempt: convert ecg signal via Wavelet-transformation into images
def preprocess_data(ecg_leads_, ecg_labels_):
    ecg_leads_list = list()
    labels_list = list()

    #categorical_to_numerical = {"N": 0, "A": 1, "O": 2, "~": 3}

    # create lists containing all data from all csv-files
    for lead in ecg_leads_:
        ecg_leads_list.append(np.array(lead).astype(np.float32))
    for label in ecg_labels_:
        labels_list.append(np.array(label))

    # convert lists to numpy arrays
    ecg_leads_array = np.array(ecg_leads_list)  # .astype(np.float32)
    ecg_labels_array = np.array(labels_list)  # .astype(np.float32)
    print("ecg_leads_array.shape \t" + str(ecg_leads_array.shape))
    print("ecg_labels_array.shape \t" + str(ecg_labels_array.shape))

    # convert categorical labels into numerical
    _, ecg_labels_array = np.unique(ecg_labels_array, return_inverse=True)  # Note: first variable _ is unused

    X_train, X_test, y_train, y_test = train_test_split(ecg_leads_array, ecg_labels_array, test_size=0.2, random_state=0)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # fills variable length tensors with zeros until all tensors have equal dimensions as he longest sequence
    X_train_tensor = pad_sequence([torch.tensor(x) for x in X_train], batch_first=True)
    X_test_tensor = pad_sequence([torch.tensor(x) for x in X_test], batch_first=True)
    # converts numpy-arrays into tensors
    y_train_tensor = tf.convert_to_tensor(np.asarray(y_train).astype(np.float32), dtype=tf.int32)
    y_test_tensor = tf.convert_to_tensor(np.asarray(y_test).astype(np.float32), dtype=tf.int32)

    X_train_tensor = np.expand_dims(X_train_tensor, -1)
    X_test_tensor = np.expand_dims(X_test_tensor, -1)
    y_train_tensor = np.expand_dims(y_train_tensor, -1)
    y_test_tensor = np.expand_dims(y_test_tensor, -1)

    print(X_train_tensor.shape, X_test_tensor.shape)
    print(y_train_tensor.shape, y_test_tensor.shape)

    return X_train_tensor, X_test_tensor , y_train_tensor, y_test_tensor

# ToDo: search in literature for suitable model architectures
# Build the model
def create_model():
    # The model architecture type is sequential hence that is used
    model = Sequential()

    # We are using 4 convolution layers for feature extraction
    model.add(Conv1D(filters=512, kernel_size=32, padding='same', kernel_initializer='normal', activation='relu', input_shape=(18000, 1)))  # (256, 2)))
    #model.add(Conv1D(filters=512, kernel_size=32, padding='same', kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))  # This is the dropout layer. It's main function is to inactivate 20% of neurons in order to prevent overfitting
    #model.add(Conv1D(filters=256, kernel_size=32, padding='same', kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Conv1D(filters=256, kernel_size=32, padding='same', kernel_initializer='normal', activation='relu'))
    model.add(MaxPool1D(pool_size=128))  # We use MaxPooling with a filter size of 128. This also contributes to generalization
    model.add(Dropout(0.2))

    # The prevous step gices an output of multi dimentional data, which cannot be fead directly into the feed forward neural network. Hence, the model is flattened
    model.add(Flatten())
    # One hidden layer of 128 neurons have been used in order to have better classification results
    model.add(Dense(units=128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    # The final neuron HAS to be 1 in number and cannot be more than that. This is because this is a binary classification problem and only 1 neuron is enough to denote the class '1' or '0'
    model.add(Dense(units=1, activation='sigmoid'))

    return model


if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig


    ecg_leads, ecg_labels, fs, ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name  # Sampling-Frequenz 300 Hz

    X_train, X_test , y_train, y_test = preprocess_data(ecg_leads, ecg_labels)

    # Create a basic model instance
    model = create_model()
    # Print the summary of the model
    model.summary()

    # Create a callback that saves the model's weights after each epoch
    checkpoint_path = "checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    start_time = time.time()

        # Train the model
    # To my experience, the Stocastic Gradient Descent Optimizer works the best. Adam optimizer also works but not as good as SGD
    optimizer = optimizers.SGD(learning_rate=0.001, momentum=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X_train, y_train,  batch_size=64, epochs=5, validation_data=(X_test, y_test),  callbacks=[cp_callback])

    pred_time = time.time() - start_time
    print("Runtime", pred_time, "s")

    #ToDo: save trained model in .npy format
    model.save('saved_model/my_model')


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