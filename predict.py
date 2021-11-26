# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""
# To filter Warnings and Information logs
# 0 | DEBUG | [Default] Print all messages
# 1 | INFO | Filter out INFO messages
# 2 | WARNING | Filter out INFO & WARNING messages
# 3 | ERROR | Filter out all messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple
import tensorflow as tf

import train
from train import load_test_images
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import preprocess_ecg_lead as prep
import plots
import glob



###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''


#------------------------------------------------------------------------------
# Euer Code ab hier

    #ToDo: Add the same preprocessing steps to predict function as in train function to ensure same data format
    # ToDo: transform ecg_leads into image data and fit it with image_data_generator into the model
    directory = '../workspace/'
    for i, (ecg_lead, ecg_name) in enumerate(zip(ecg_leads, ecg_names)):
        # segment ecg lead into segments containing 3 r-peaks each
        ecg_segments = prep.segment_ecg_lead(ecg_lead, fs)
        # convert arrays of segmented data into images and save them in working directory
        test_image_directory = prep.segment_to_test_img(ecg_segments, ecg_name, directory)

    # load generated images of 3 r-peak ecg segments
    test_generator = load_test_images(directory)        # labelt die Bilder noch falsch. ToDo: Bilder ohne Labels laden (ohne Image-Data_generator)

    #labels = ['~', 'A', 'N', 'O']
    model = prep.load_model_from_name(model_name)
    # Generate predictions for samples
    predictions = list()
    predictions = model.predict(test_generator)  # , num_of_test_samples // batch_size+1),





    # Confution Matrix and Classification Report
    predicted_categories = tf.argmax(predictions, axis=1)
    #cm = confusion_matrix(test_generator.classes, predicted_categories)
    #print('Confusion Matrix')
    #print(cm)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)  # , display_labels=labels)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.show()
    #print('Classification Report')
    #print(classification_report(test_generator.classes, predicted_categories))  # , target_names=labels))




    # ToDo: predicitons in richtiges Format bringen
    '''
    #assert isinstance(predictions[0], tuple), \
    #AssertionError: Elemente der Liste predictions muss ein Tuple sein aber <class 'list'> gegenen.
    '''

    predictions = predictions.tolist()
    #------------------------------------------------------------------------------
    return predictions  # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
                               
        
